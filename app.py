# 실행: python -m streamlit run app.py
# 필수 설치: pip install deepface tf-keras transformers torch torchvision pillow

import streamlit as st
import tempfile
import os
import json
import zipfile
import io
from google import genai
from google.genai import types

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 라이브러리 임포트 ──────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import open_clip
    import torch
    from PIL import Image
    import torch.nn.functional as F
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# tokenizer는 CLIP 로딩 후 별도 보관
_clip_tokenizer = None


# ── API 클라이언트 ─────────────────────────────────────────────
client = genai.Client(
    api_key=st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
)

# ── 세션 초기화 ────────────────────────────────────────────────
defaults = {
    'stage': 'input',
    'character_image': None,
    'storyboard_data': [],
    'sample_data': [],
    'topic': '',
    'char_name': '',
    'char_description': '',
    'act_plan': None,
    'scenes_by_act': None,      # {act: [{text, type, camera, duration, reason}]}
    'df_threshold': 70,  # Facenet512 cosine 공식 기준 (distance<0.30 ↔ score>70)
    'clip_threshold': 50,       # image-image cosine 정규화: (cosine-0.5)/0.5×100
    'clip_text_threshold': 25,  # text-image cosine 정규화: 낮은 기준 (분포 상이)
    'max_retries': 3,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="AI 콘티 제작소", layout="wide")
st.title("🎬 단편 영화 AI 콘티 제작소")

STAGES = ['input', 'character', 'scenario', 'sample', 'storyboard']
stage_labels = ['① 캐릭터 설정', '② 3면도 확인', '③ 시나리오 확인', '④ 샘플 테스트', '⑤ 스토리보드']
current_idx = STAGES.index(st.session_state.stage)
st.progress(current_idx / (len(STAGES) - 1), text=stage_labels[current_idx])
st.divider()


# ══════════════════════════════════════════════════════════════
# [헬퍼] 모델 로딩 (캐시)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    return model, preprocess, tokenizer


def bytes_to_pil(img_bytes):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")


# ══════════════════════════════════════════════════════════════
# [헬퍼] 씬 유형별 검수 전략
# ══════════════════════════════════════════════════════════════
SCENE_TYPE_CONFIG = {
    "face_visible": {
        "label": "얼굴 노출",
        "emoji": "👤",
        "use_deepface": True,
        "use_clip": True,
        "use_clip_text": False,
        "correction_strategy": "face_focus",
    },

    "face_hidden": {
        "label": "얼굴 미노출",
        "emoji": "🎭",
        "use_deepface": False,
        "use_clip": True,
        "use_clip_text": True,
        "correction_strategy": "body_focus",
    },

    "crowd": {
        "label": "군중씬",
        "emoji": "👥",
        "use_deepface": False,
        "use_clip": False,
        "use_clip_text": True,
        "correction_strategy": "composition_focus",
    },
}


# ══════════════════════════════════════════════════════════════
# [헬퍼] 검수 함수들
# ══════════════════════════════════════════════════════════════
def check_deepface(ref_bytes, gen_bytes):
    result = {"deepface_score": -1, "deepface_passed": True, "deepface_reason": "미실행"}
    if not DEEPFACE_AVAILABLE:
        result["deepface_reason"] = "미설치"
        return result
    ref_path = gen_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
            f1.write(ref_bytes); ref_path = f1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            f2.write(gen_bytes); gen_path = f2.name
        df = DeepFace.verify(img1_path=ref_path, img2_path=gen_path,
                             model_name="Facenet512", detector_backend="retinaface",
                             distance_metric="cosine", enforce_detection=False)
        dist = df.get("distance", 1.0)
        score = round(max(0.0, (1.0 - dist) * 100), 1)
        result["deepface_score"] = score
        result["deepface_passed"] = df.get("verified", False)
        result["deepface_reason"] = f"{score}점 (distance={dist:.3f})"
    except Exception as e:
        err = str(e)
        result["deepface_reason"] = "얼굴 미검출" if "face" in err.lower() else f"오류: {err}"
    finally:
        for p in [ref_path, gen_path]:
            try: os.unlink(p) if p else None
            except: pass
    return result

def check_clip_image(ref_bytes, gen_bytes):
    """CLIP image-image 유사도 (캐릭터 외형 일관성)"""
    result = {
        "clip_score": -1,
        "clip_passed": False,
        "clip_reason": "미실행"
    }
    if not CLIP_AVAILABLE:
        result["clip_reason"] = "미설치"
        return result
    try:
        model, preprocess, _ = load_clip_model()
        device = "cpu"
        model = model.to(device)

        img1 = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(gen_bytes)).convert("RGB")
        img1 = preprocess(img1).unsqueeze(0).to(device)
        img2 = preprocess(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = model.encode_image(img1)
            feat2 = model.encode_image(img2)

        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        sim = (feat1 @ feat2.T).item()
        score = round(sim * 100, 1)

        result["clip_score"] = score
        result["clip_passed"] = True
        result["clip_reason"] = f"{score}점"
    except Exception as e:
        result["clip_reason"] = str(e)
    return result


def check_clip_text(gen_bytes, scene_text):
    """CLIP text-image 유사도 (씬 묘사 부합도)"""
    result = {
        "clip_text_score": -1,
        "clip_text_passed": False,
        "clip_text_reason": "미실행"
    }
    if not CLIP_AVAILABLE:
        result["clip_text_reason"] = "미설치"
        return result
    try:
        model, preprocess, tokenizer = load_clip_model()
        device = "cpu"
        model = model.to(device)

        img = Image.open(io.BytesIO(gen_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # 씬 텍스트 토크나이즈 (512 토큰 초과 방지 위해 앞 77토큰만 사용)
        text_tokens = tokenizer([scene_text[:200]]).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(text_tokens)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sim = (img_feat @ txt_feat.T).item()
        # text-image cosine은 보통 0.1~0.35 범위 → 0~100 정규화 (0.0→0, 0.5→100)
        score = round(max(0.0, sim * 200), 1)

        result["clip_text_score"] = score
        result["clip_text_passed"] = True
        result["clip_text_reason"] = f"{score}점 (cosine={sim:.3f})"
    except Exception as e:
        result["clip_text_reason"] = str(e)
    return result


def run_adaptive_check(ref_bytes, gen_bytes, scene_text, scene_type,
                       df_thr, clip_thr, clip_text_thr):
    """
    씬 유형에 따라 적용할 모델을 결정하고 통과 여부 판정.
    - face_visible : DeepFace + CLIP image-image
    - face_hidden  : CLIP image-image + CLIP text-image
    - crowd        : CLIP text-image 전용
    """
    cfg = SCENE_TYPE_CONFIG.get(scene_type, SCENE_TYPE_CONFIG["face_visible"])

    df_result  = check_deepface(ref_bytes, gen_bytes)
    cl_result  = check_clip_image(ref_bytes, gen_bytes)
    ct_result  = check_clip_text(gen_bytes, scene_text)

    df_s  = df_result["deepface_score"]
    cl_s  = cl_result["clip_score"]
    ct_s  = ct_result["clip_text_score"]

    conditions = []
    fail_reasons = []

    if cfg["use_deepface"]:
        df_ok = (df_s == -1 or df_s >= df_thr)
        conditions.append(df_ok)
        if df_s != -1 and not df_ok:
            fail_reasons.append(f"DeepFace 미달 ({df_s}점 < {df_thr}점)")

    if cfg["use_clip"]:
        cl_ok = (cl_s == -1 or cl_s >= clip_thr)
        conditions.append(cl_ok)
        if cl_s != -1 and not cl_ok:
            fail_reasons.append(f"CLIP image 미달 ({cl_s}점 < {clip_thr}점)")

    if cfg["use_clip_text"]:
        ct_ok = (ct_s == -1 or ct_s >= clip_text_thr)
        conditions.append(ct_ok)
        if ct_s != -1 and not ct_ok:
            fail_reasons.append(f"CLIP text 미달 ({ct_s}점 < {clip_text_thr}점)")

    overall = all(conditions) if conditions else True
    fail = " | ".join(fail_reasons) if fail_reasons else None

    active_models = []
    if cfg["use_deepface"]:    active_models.append("deepface")
    if cfg["use_clip"]:        active_models.append("clip")
    if cfg["use_clip_text"]:   active_models.append("clip_text")

    return {
        **df_result, **cl_result, **ct_result,
        "overall_passed": overall,
        "fail_reason": fail,
        "scene_type": scene_type,
        "active_models": active_models,
    }

    df_s = df_result["deepface_score"]
    cl_s = cl_result["clip_score"]



# ══════════════════════════════════════════════════════════════
# [헬퍼] 씬 유형별 Self-Correction 프롬프트
# ══════════════════════════════════════════════════════════════
def build_prompt(name, desc, scene_text, scene_type, prev_score=-1, attempt=1):
    cfg = SCENE_TYPE_CONFIG.get(scene_type, SCENE_TYPE_CONFIG["face_visible"])
    strategy = cfg["correction_strategy"]

    base = (
        f"You are a cinematic image generator. "
        f"The character in the reference image is named {name}. "
        f"Generate a hyperrealistic cinematic film still (16:9 widescreen). "
        f"Scene: {scene_text}. "
        f"CRITICAL: Keep {name}'s costume, hair, and overall appearance EXACTLY identical to reference. "
        "Shot on ARRI Alexa 65, anamorphic lens, 8K, dramatic cinematic lighting."
    )

    if attempt == 1 or prev_score < 0:
        # 씬 유형별 초기 프롬프트 지시
        if scene_type == "face_visible":
            return base + (
                f" Ensure {name}'s face is CLEARLY VISIBLE and in SHARP FOCUS. "
                "Use medium shot or closer. Face must match reference exactly."
            )
        elif scene_type == "face_hidden":
            return base + (
                f" The scene may show {name} from behind or at distance — this is intentional. "
                f"Focus on consistent body shape, costume, and silhouette of {name}. "
                "Do NOT force a face reveal if the scene doesn't call for it."
            )
        else:  # crowd
            return base + (
                f" Focus on overall composition and atmosphere. "
                f"{name} may be one of many figures. Prioritize scene fidelity over face visibility."
            )

    # Self-Correction: 전략별 분기
    if strategy == "face_focus":
        return (
            f"Self-correction attempt #{attempt}. Previous face similarity: {prev_score:.1f}/100 (FAILED). "
            f"Strategy: Show {name}'s face MORE clearly. "
            "Use closer framing (medium-close or close-up). Ensure face is front-facing and well-lit. "
            f"Scene context: {scene_text}. 16:9. Face must match reference exactly."
        )
    elif strategy == "body_focus":
        return (
            f"Self-correction attempt #{attempt}. Previous body/costume similarity: {prev_score:.1f}/100 (FAILED). "
            f"Strategy: Emphasize {name}'s costume, body shape, and silhouette consistency. "
            "Do NOT force face visibility — maintain the intended camera angle. "
            f"Scene: {scene_text}. Costume and body proportions must match reference exactly."
        )
    else:  # composition_focus
        return (
            f"Self-correction attempt #{attempt}. Previous composition similarity: {prev_score:.1f}/100 (FAILED). "
            f"Strategy: Improve overall scene composition and atmosphere consistency. "
            f"Scene: {scene_text}. Maintain visual style and lighting from reference."
        )


def generate_image(name, desc, scene_text, scene_type, ref_bytes, prev_score=-1, attempt=1):
    prompt = build_prompt(name, desc, scene_text, scene_type, prev_score, attempt)
    resp = client.models.generate_content(
        model='gemini-2.5-flash-image',
        contents=[
            types.Part.from_bytes(data=ref_bytes, mime_type='image/png'),
            types.Part.from_text(text=prompt)
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9")
        )
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    return None


# ══════════════════════════════════════════════════════════════
# [헬퍼] 씬 1개 생성 + 적응형 검수 루프
# ══════════════════════════════════════════════════════════════
def generate_and_check(name, desc, scene_info, ref_bytes,
                       df_thr, clip_thr, clip_text_thr, max_retries,
                       label, status_box):
    scene_text = scene_info.get("text", "")
    scene_type = scene_info.get("type", "face_visible")
    cfg = SCENE_TYPE_CONFIG.get(scene_type, SCENE_TYPE_CONFIG["face_visible"])

    best_image = None
    best_scores = {}
    passed = False
    all_attempts = []
    prev_score = -1

    for attempt in range(1, max_retries + 1):
        type_tag = f"[{cfg['emoji']} {cfg['label']}]"
        if attempt == 1:
            status_box.info(f"🎨 **{label}** {type_tag} — 생성 중...")
        else:
            status_box.info(
                f"🔄 **{label}** {type_tag} — {attempt}차 Self-Correction "
                f"({cfg['correction_strategy']} 전략 | 이전: {prev_score:.1f}점)"
            )

        try:
            image_bytes = generate_image(
                name, desc, scene_text, scene_type, ref_bytes, prev_score, attempt
            )
        except Exception as e:
            status_box.error(f"생성 오류: {e}")
            break

        if not image_bytes:
            status_box.warning("이미지 응답 없음, 재시도...")
            continue

        check = run_adaptive_check(
            ref_bytes, image_bytes, scene_text, scene_type,
            df_thr, clip_thr, clip_text_thr
        )
        attempt_passed = check["overall_passed"]
        fail_reason = check.get("fail_reason")

        # Self-Correction용 기준 점수 (씬 유형별 주력 모델 기준)
        if scene_type == "face_visible":
            active_score = check.get("deepface_score", -1)
        elif scene_type == "face_hidden":
            active_score = check.get("clip_score", -1)
        else:  # crowd
            active_score = check.get("clip_text_score", -1)

        if active_score != -1:
            prev_score = active_score

        if best_image is None or active_score > best_scores.get("active", -1):
            best_image = image_bytes
            best_scores = {
                "active": active_score,
                "deepface": check.get("deepface_score", -1),
                "clip": check.get("clip_score", -1),
                "clip_text": check.get("clip_text_score", -1),
            }

        all_attempts.append({
            "attempt": attempt,
            "image_bytes": image_bytes,
            "check": check,
            "passed": attempt_passed,
            "fail_reason": fail_reason,
            "self_corrected": attempt > 1,
            "strategy": cfg["correction_strategy"],
        })

        if attempt_passed:
            passed = True
            active_models = check.get("active_models", [])
            score_parts = []
            if "deepface"  in active_models: score_parts.append(f"DeepFace:{check.get('deepface_score',-1)}점")
            if "clip"      in active_models: score_parts.append(f"CLIP:{check.get('clip_score',-1)}점")
            if "clip_text" in active_models: score_parts.append(f"CLIP-Text:{check.get('clip_text_score',-1)}점")
            status_box.success(f"✅ **{label}** 통과! ({' | '.join(score_parts)})")
            break
        else:
            status_box.warning(f"⚠️ **{label}** {attempt}차 탈락 — {fail_reason}")

    if not passed and best_image:
        status_box.warning(
            f"⚠️ **{label}** 최고점 채택 "
            f"(DeepFace:{best_scores.get('deepface',-1)} / "
            f"CLIP:{best_scores.get('clip',-1)} / "
            f"CLIP-Text:{best_scores.get('clip_text',-1)})"
        )
    elif not best_image:
        status_box.error(f"❌ **{label}** 이미지 생성 실패 — 모든 시도에서 이미지를 받지 못했습니다.")

    return {
        "image_bytes": best_image,
        "passed": passed,
        "best_scores": best_scores,
        "all_attempts": all_attempts,
        "scene_type": scene_type,
        "failed": best_image is None,
    }


# ══════════════════════════════════════════════════════════════
# [헬퍼] 점수 배지
# ══════════════════════════════════════════════════════════════
def score_badge(score, label, threshold=60, active=True):
    if not active: return f"⚪ {label}: (미적용)"
    if score == -1: return f"🔵 {label}: N/A"
    color = "🟢" if score >= threshold else "🔴"
    return f"{color} {label}: {score}점"


# ══════════════════════════════════════════════════════════════
# [헬퍼] ZIP 생성 (이미지)
# ══════════════════════════════════════════════════════════════
def make_image_zip(storyboard_data):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in storyboard_data:
            act = scene.get("act", "unknown")
            cut = scene.get("cut_num", 0)
            zf.writestr(f"{act}_{cut:02d}컷.png", scene["image_bytes"])
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# [헬퍼] 시나리오 TXT 생성
# ══════════════════════════════════════════════════════════════
def make_scenario_txt(char_name, act_plan, scenes_by_act, storyboard_data):
    lines = []
    lines.append("=" * 60)
    lines.append("단편 영화 AI 콘티 제작소 — 시나리오 & 스토리보드")
    lines.append("=" * 60)
    lines.append(f"캐릭터: {char_name}")

    if act_plan:
        lines.append(f"장르: {act_plan.get('genre', 'N/A')}")
        lines.append(f"속도감: {act_plan.get('pacing', 'N/A')}")
        lines.append(f"분석 근거: {act_plan.get('reasoning', 'N/A')}")
        total_dur = act_plan.get('total_duration_sec', 300)
        lines.append(f"예상 총 길이: {total_dur}초 ({total_dur//60}분 {total_dur%60}초)")
    lines.append("")

    ACT_CONFIG = [("1막", "설정 Setup", "🌅"), ("2막", "대립 Confrontation", "⚡"), ("3막", "해결 Resolution", "🌟")]

    for act_kr, act_label, emoji in ACT_CONFIG:
        scenes_list = scenes_by_act.get(act_kr, [])
        if not scenes_list:
            continue

        act_dur = sum(s.get("duration", 0) for s in scenes_list)
        lines.append(f"{emoji} {act_kr} — {act_label}  (총 {act_dur}초)")
        lines.append("-" * 40)

        for j, scene_info in enumerate(scenes_list):
            s_type = scene_info.get("type", "face_visible")
            cfg = SCENE_TYPE_CONFIG.get(s_type, {})
            type_label = cfg.get("label", s_type)
            camera = scene_info.get("camera", "")
            duration = scene_info.get("duration", 0)
            reason = scene_info.get("reason", "")

            # 검수 결과 있으면 같이 기록
            score_line = ""
            for sb in storyboard_data:
                if sb.get("act") == act_kr and sb.get("cut_num") == j + 1:
                    sc = sb.get("best_scores", {})
                    score_line = (
                        f"  → 검수: DeepFace {sc.get('deepface',-1)}점 | "
                        f"CLIP {sc.get('clip',-1)}점 | CLIP-Text {sc.get('clip_text',-1)}점 | "
                        f"{'✅ 통과' if sb.get('passed') else '⚠️ 최고점 채택'}"
                    )
                    break

            lines.append(f"  씬{j+1} [{type_label} / {camera} / {duration}초]")
            lines.append(f"  {scene_info.get('text', '')}")
            if reason:
                lines.append(f"  (분류 근거: {reason})")
            if score_line:
                lines.append(score_line)
            lines.append("")

        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# [단계 0] 캐릭터 설정
# ══════════════════════════════════════════════════════════════
if st.session_state.stage == 'input':
    st.subheader("🧑‍🎨 주인공 외형 설정")

    avail = []
    avail.append("✅ DeepFace" if DEEPFACE_AVAILABLE else "❌ DeepFace")
    avail.append("✅ CLIP" if CLIP_AVAILABLE else "❌ CLIP")
    st.info("검수 모델: " + " | ".join(avail))

    topic = st.text_input("주인공 외형/연출 의도",
                          value=st.session_state.topic,
                          placeholder="예: 플라스틱 질감의 마네킹 같은 얼굴을 한 사이버펑크 캐릭터")
    char_name = st.text_input("캐릭터 이름 (영문 권장)",
                              value=st.session_state.char_name,
                              placeholder="예: NOVA, MIRA, ZERO ...")

    if st.button("캐릭터 3면도 생성하기", use_container_width=True):
        if not topic.strip() or not char_name.strip():
            st.warning("모든 항목을 입력해 주세요.")
        else:
            st.session_state.topic = topic
            st.session_state.char_name = char_name.strip().upper()
            with st.spinner("외형 키워드 분석 중..."):
                try:
                    resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"Summarize into English keywords (max 20 words) for image prompts. No sentences, no quotes.\nDescription: {topic}"
                    )
                    st.session_state.char_description = resp.text.strip()
                except Exception:
                    st.session_state.char_description = topic

            with st.spinner("3면도 생성 중..."):
                try:
                    name = st.session_state.char_name
                    desc = st.session_state.char_description
                    turnaround_prompt = (
                        f"Professional 3D character design turnaround sheet of {name}. "
                        f"Character appearance: {desc}. "
                        "THREE VIEWS side by side in a single image: "
                        "LEFT: full-body FRONT VIEW facing camera directly. "
                        "CENTER: full-body SIDE VIEW (90 degree profile, facing right). "
                        "RIGHT: full-body BACK VIEW facing away from camera. "
                        "All three views show the EXACT SAME character with identical costume, proportions, colors, and details. "
                        "Pure solid white background. No shadows on background. "
                        "Full body visible from head to toe in each view. "
                        "Character centered and same height in all three panels. "
                        "Clean separation between the three views. "
                        "High quality 3D animation style, cute stylized character, clean Blender 3D render, soft studio lighting, character reference sheet style."
                    )
                    result = client.models.generate_images(
                        model='imagen-4.0-generate-001',
                        prompt=turnaround_prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1, aspect_ratio="16:9",
                            person_generation="ALLOW_ADULT")
                    )
                    st.session_state.character_image = result.generated_images[0].image.image_bytes
                    st.session_state.stage = 'character'
                    st.rerun()
                except Exception as e:
                    st.error(f"생성 오류: {e}")


# ══════════════════════════════════════════════════════════════
# [단계 1] 3면도 확인
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'character':
    name = st.session_state.char_name
    st.subheader(f"🧑‍🎨 [{name}] 3면도 확인")
    st.image(st.session_state.character_image, caption=name)
    st.caption(f"외형 키워드: {st.session_state.char_description}")
    st.info("확정하면 이후 모든 씬에서 이 외형이 유지됩니다.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 재생성", use_container_width=True):
            st.session_state.stage = 'input'
            st.rerun()
    with col2:
        if st.button("✅ 확정! 시나리오 작성으로", use_container_width=True):
            st.session_state.stage = 'scenario'
            st.rerun()


# ══════════════════════════════════════════════════════════════
# [단계 2] 시나리오 생성 + 확인/수정
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'scenario':
    name = st.session_state.char_name
    desc = st.session_state.char_description
    st.subheader(f"📝 [{name}]의 시나리오 작성")

    with st.expander("📌 캐릭터 레퍼런스"):
        st.image(st.session_state.character_image, width=250)

    plot_input = st.text_area(
        "단편 영화 줄거리 (5분 분량 기준)",
        placeholder="예: 버려진 공장에서 깨어난 마네킹이 자신이 플라스틱으로 만들어졌다는 사실을 깨닫고 인간이 되고자 탈출을 시도한다."
    )

    if st.button("🎬 시나리오 자동 생성", use_container_width=True):
        if not plot_input.strip():
            st.warning("줄거리를 입력해 주세요.")
        else:
            # ── 1단계: 장르/속도감 분석 → 씬 수 + 예상 시간 결정 ──
            with st.spinner("장르 및 속도감 분석 중..."):
                try:
                    plan_resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"""
당신은 애니메이션 편집 전문가입니다. 아래 줄거리를 분석하여 약 30~50초 분량의 핵심 시퀀스 스토리보드를 기획하세요.

[분석 기준]
- 애니메이션/영화의 실제 컷(Cut) 지속시간은 매우 짧습니다.
- 장르별 평균 컷(Cut) 지속시간:
  * 액션/스릴러: 컷당 1~3초 (빠른 전환)
  * 일상/드라마: 컷당 3~6초 (표정 및 대사 전달)
  * 설명/풍경: 컷당 5~8초 (상황 전달)

- 3막 구조 컷 수 배분 (총 10컷 내외로 구성):
  * 1막(설정): 2~3컷
  * 2막(대립): 4~6컷
  * 3막(해결): 2~3컷

[출력 형식] 반드시 아래 JSON만 출력하세요. 다른 텍스트 없이:
{{
  "genre": "장르명",
  "pacing": "fast/medium/slow",
  "avg_scene_duration": 평균컷길이(초),
  "total_duration_sec": 전체합산시간(초),
  "reasoning": "컷 수 결정 근거 (한국어, 2문장 이내)",
  "act1_scenes": N,
  "act1_duration": 초,
  "act2_scenes": N,
  "act2_duration": 초,
  "act3_scenes": N,
  "act3_duration": 초
}}

줄거리: {plot_input}
"""
                    )
                    raw = plan_resp.text.strip().replace("```json","").replace("```","").strip()
                    act_plan = json.loads(raw)
                    st.session_state.act_plan = act_plan
                    act1_n = int(act_plan.get("act1_scenes", 3))
                    act2_n = int(act_plan.get("act2_scenes", 3))
                    act3_n = int(act_plan.get("act3_scenes", 3))
                except Exception as e:
                    st.warning(f"분석 실패, 기본값 사용: {e}")
                    act1_n, act2_n, act3_n = 2, 4, 2
                    st.session_state.act_plan = None

            # ── 2단계: 씬 유형 분류 포함 시나리오 생성 ──────────────
            with st.spinner("시나리오 + 씬 유형 분류 중..."):
                try:
                    scene_resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"""
당신은 단편 영화 시나리오 작가이자 콘티 감독입니다.
주인공 이름: {name}, 외형: {desc}

아래 줄거리를 3막 구조로 확장하고 각 씬을 상세히 작성하세요.

[씬 수 요구사항]
- 1막(설정): 정확히 {act1_n}개 씬
- 2막(대립): 정확히 {act2_n}개 씬
- 3막(해결): 정확히 {act3_n}개 씬

[각 씬 작성 기준]
- 씬 묘사: 카메라 앵글, 조명, {name}의 행동과 감정, 배경을 구체적으로 기술 (2~3문장)
- 씬 유형 분류:
  * face_visible: {name}의 얼굴이 명확히 보이는 씬 (정면, 클로즈업, 미디엄샷)
  * face_hidden: 뒷모습, 원거리, 측면 등 얼굴이 안 보이거나 부분적으로 보이는 씬
  * crowd: 군중 속 씬 또는 {name}이 작게 등장하는 와이드샷
- 카메라: close-up / medium / wide / over-shoulder / aerial 중 선택
- 예상 지속시간(초): 씬의 호흡에 맞게 설정

[출력 형식] 반드시 아래 JSON 배열만 출력하세요. 다른 텍스트 없이:
[
  {{"act":"1막","text":"1막_씬1: [한국어 씬 묘사]","type":"face_visible","camera":"medium","duration":25,"reason":"캐릭터 첫 등장, 얼굴 인식 필요"}},
  {{"act":"1막","text":"1막_씬2: [한국어 씬 묘사]","type":"face_hidden","camera":"wide","duration":30,"reason":"공간 설명 씬"}},
  ...
]

줄거리: {plot_input}
"""
                    )
                    raw = scene_resp.text.strip().replace("```json","").replace("```","").strip()
                    scenes_flat = json.loads(raw)

                    # act별로 그룹핑
                    scenes_by_act = {"1막": [], "2막": [], "3막": []}
                    for s in scenes_flat:
                        act = s.get("act", "1막")
                        if act in scenes_by_act:
                            scenes_by_act[act].append(s)

                    st.session_state.scenes_by_act = scenes_by_act
                    st.rerun()

                except Exception as e:
                    st.error(f"시나리오 생성 오류: {e}")

    # ── 생성된 시나리오 표시 + 수정 ────────────────────────────
    if st.session_state.scenes_by_act:
        plan = st.session_state.act_plan
        total_scenes = sum(len(v) for v in st.session_state.scenes_by_act.values())

        if plan:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("장르", plan.get('genre', 'N/A'))
            col2.metric("속도감", plan.get('pacing', 'N/A'))
            col3.metric("총 씬 수", f"{total_scenes}컷")
            col4.metric("평균 씬 길이", f"{plan.get('avg_scene_duration', 'N/A')}초")
            st.caption(f"📌 {plan.get('reasoning', '')}")

        st.markdown("### 📖 생성된 시나리오 — 직접 수정 가능합니다")
        st.info("💡 씬 내용을 수정하고 유형/카메라도 변경할 수 있습니다. 확정 후 샘플 테스트로 넘어갑니다.")

        ACT_EMOJI = {"1막": "🌅", "2막": "⚡", "3막": "🌟"}
        ACT_DESC  = {"1막": "설정", "2막": "대립", "3막": "해결"}
        TYPE_OPTIONS = ["face_visible", "face_hidden", "crowd"]
        CAMERA_OPTIONS = ["close-up", "medium", "wide", "over-shoulder", "aerial"]

        edited_scenes = {"1막": [], "2막": [], "3막": []}

        for act_kr in ["1막", "2막", "3막"]:
            scenes_list = st.session_state.scenes_by_act.get(act_kr, [])
            if not scenes_list:
                continue

            act_dur = sum(s.get("duration", 0) for s in scenes_list)
            st.markdown(f"#### {ACT_EMOJI[act_kr]} {act_kr} — {ACT_DESC[act_kr]} ({len(scenes_list)}컷 / 약 {act_dur}초)")

            for j, scene_info in enumerate(scenes_list):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([5, 2, 2])
                    with c1:
                        edited_text = st.text_area(
                            f"{act_kr} 씬{j+1}",
                            value=scene_info.get("text", ""),
                            key=f"text_{act_kr}_{j}",
                            height=90
                        )
                    with c2:
                        edited_type = st.selectbox(
                            "씬 유형",
                            TYPE_OPTIONS,
                            index=TYPE_OPTIONS.index(scene_info.get("type", "face_visible")),
                            key=f"type_{act_kr}_{j}",
                            format_func=lambda x: f"{SCENE_TYPE_CONFIG[x]['emoji']} {SCENE_TYPE_CONFIG[x]['label']}"
                        )
                        edited_camera = st.selectbox(
                            "카메라",
                            CAMERA_OPTIONS,
                            index=CAMERA_OPTIONS.index(scene_info.get("camera", "medium"))
                            if scene_info.get("camera") in CAMERA_OPTIONS else 0,
                            key=f"camera_{act_kr}_{j}"
                        )
                    with c3:
                        edited_dur = st.number_input(
                            "지속시간(초)",
                            min_value=5, max_value=120,
                            value=int(scene_info.get("duration", 20)),
                            key=f"dur_{act_kr}_{j}"
                        )
                        cfg = SCENE_TYPE_CONFIG.get(edited_type, {})
                    st.caption(
                            f"적용 모델:\n"
                            f"{'✅' if cfg.get('use_deepface') else '⬜'} DeepFace\n"
                            f"{'✅' if cfg.get('use_clip') else '⬜'} CLIP image\n"
                            f"{'✅' if cfg.get('use_clip_text') else '⬜'} CLIP text\n"
                        )

                    edited_scenes[act_kr].append({
                        "text": edited_text,
                        "type": edited_type,
                        "camera": edited_camera,
                        "duration": edited_dur,
                        "reason": scene_info.get("reason", ""),
                    })

            st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 시나리오 다시 생성", use_container_width=True):
                st.session_state.scenes_by_act = None
                st.rerun()
        with col2:
            if st.button("✅ 확정 → 샘플 테스트", use_container_width=True):
                st.session_state.scenes_by_act = edited_scenes
                st.session_state.stage = 'sample'
                st.rerun()


# ══════════════════════════════════════════════════════════════
# [단계 3] 샘플 테스트
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'sample':
    name = st.session_state.char_name
    desc = st.session_state.char_description
    scenes_by_act = st.session_state.scenes_by_act
    ref_bytes = st.session_state.character_image

    st.subheader("🧪 샘플 테스트 (각 막 첫 번째 씬)")
    st.info("전체 생성 전에 각 막 첫 씬만 먼저 생성해서 검수 설정을 확인합니다.")

    with st.expander("⚙️ 검수 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            df_thr = st.slider("DeepFace 기준점", 30, 90, st.session_state.df_threshold, 5,
                               help="face_visible 씬에만 적용. 기본값 70점 = Facenet512 공식 동일인물 판정 기준 (cosine distance < 0.30)")
        with col2:
            clip_thr = st.slider("CLIP image 기준점", 0, 100, st.session_state.clip_threshold, 5,
                                 help="face_visible + face_hidden 씬에 적용 (캐릭터 외형 일관성)")
        with col3:
            clip_text_thr = st.slider("CLIP text 기준점", 0, 100, st.session_state.clip_text_threshold, 5,
                                      help="face_hidden + crowd 씬에 적용 (씬 묘사 부합도)")
        max_retries = st.slider("최대 재시도", 1, 5, st.session_state.max_retries)

        st.session_state.df_threshold = df_thr
        st.session_state.clip_threshold = clip_thr
        st.session_state.clip_text_threshold = clip_text_thr
        st.session_state.max_retries = max_retries

    if st.button("🧪 샘플 3컷 생성", use_container_width=True):
        sample_data = []
        progress = st.progress(0)

        for act_i, act_kr in enumerate(["1막", "2막", "3막"]):
            scenes_list = scenes_by_act.get(act_kr, [])
            if not scenes_list:
                continue
            sample_scene_info = scenes_list[0]
            label = f"{act_kr} 샘플"
            status_box = st.empty()

            result = generate_and_check(
                name, desc, sample_scene_info, ref_bytes,
                df_thr, clip_thr, clip_text_thr, max_retries,
                label, status_box
            )
            result["act"] = act_kr
            result["cut_num"] = 1
            result["desc"] = sample_scene_info.get("text", "")
            result["scene_info"] = sample_scene_info
            sample_data.append(result)
            progress.progress((act_i + 1) / 3)

        st.session_state.sample_data = sample_data
        st.rerun()

    if st.session_state.sample_data:
        st.markdown("### 🖼️ 샘플 결과")
        cols = st.columns(3)
        for i, sample in enumerate(st.session_state.sample_data):
            with cols[i]:
                act_kr = sample.get("act", f"{i+1}막")
                scores = sample.get("best_scores", {})
                scene_type = sample.get("scene_type", "face_visible")
                cfg = SCENE_TYPE_CONFIG.get(scene_type, {})

                if sample.get("image_bytes") is None:
                    st.error(f"❌ {act_kr} 이미지 생성 실패")
                else:
                    st.image(sample["image_bytes"], caption=f"{act_kr} 샘플")
                st.caption(f"{cfg.get('emoji','')} {cfg.get('label', scene_type)}")
                st.caption(score_badge(scores.get("deepface",-1), "DeepFace", df_thr, cfg.get("use_deepface",True)))
                st.caption(score_badge(scores.get("clip",-1), "CLIP image", clip_thr, cfg.get("use_clip",True)))
                st.caption(score_badge(scores.get("clip_text",-1), "CLIP text", clip_text_thr, cfg.get("use_clip_text",False)))
                if sample.get("failed"):
                    st.caption("❌ 생성 실패")
                else:
                    st.caption("✅ 통과" if sample.get("passed") else "⚠️ 최고점 채택")

                failed = [a for a in sample.get("all_attempts",[]) if not a["passed"]]
                if failed:
                    with st.expander(f"🔬 탈락 {len(failed)}장"):
                        st.image(ref_bytes, width=150, caption="레퍼런스")
                        for a in sample.get("all_attempts",[]):
                            ch = a["check"]
                            sc_tag = f" 🔧{a.get('strategy','')}" if a.get("self_corrected") else ""
                            st.image(a["image_bytes"],
                                     caption=f"{a['attempt']}차{sc_tag} | {'🟢통과' if a['passed'] else '🔴탈락'}",
                                     use_container_width=True)
                            st.caption(
                                f"DeepFace:{ch.get('deepface_score',-1)} | "
                                f"CLIP:{ch.get('clip_score',-1)} | "
                                f"CLIP-Text:{ch.get('clip_text_score',-1)}"
                            )
                            if a.get("fail_reason"):
                                st.caption(f"탈락: {a['fail_reason']}")

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📝 시나리오부터 다시", use_container_width=True):
                st.session_state.stage = 'scenario'
                st.session_state.sample_data = []
                st.rerun()
        with col2:
            if st.button("🔧 설정 변경 후 재생성", use_container_width=True):
                st.session_state.sample_data = []
                st.rerun()
        with col3:
            if st.button("✅ 전체 스토리보드 생성", use_container_width=True):
                st.session_state.stage = 'storyboard'
                st.session_state.storyboard_data = []
                st.rerun()


# ══════════════════════════════════════════════════════════════
# [단계 4] 전체 스토리보드 생성 + 결과
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'storyboard':
    name = st.session_state.char_name
    desc = st.session_state.char_description
    scenes_by_act = st.session_state.scenes_by_act
    ref_bytes = st.session_state.character_image
    df_thr  = st.session_state.df_threshold
    clip_thr = st.session_state.clip_threshold
    clip_text_thr = st.session_state.clip_text_threshold
    max_retries = st.session_state.max_retries

    # ── 아직 생성 안 됐으면 생성 시작 ─────────────────────────
    if not st.session_state.storyboard_data:
        st.subheader(f"🎬 [{name}] 전체 스토리보드 생성 중...")
        st.caption(f"DeepFace / CLIP image / CLIP text | 재시도: {max_retries}회 | 씬 유형별 자동 적용")

        total_scenes = sum(len(v) for v in scenes_by_act.values())
        progress_bar = st.progress(0)
        scene_counter = 0

        for act_kr in ["1막", "2막", "3막"]:
            for cut_idx, scene_info in enumerate(scenes_by_act.get(act_kr, [])):
                cut_num = cut_idx + 1
                label = f"{act_kr} {cut_num}컷"
                status_box = st.empty()

                result = generate_and_check(
                    name, desc, scene_info, ref_bytes,
                    df_thr, clip_thr, clip_text_thr, max_retries,
                    label, status_box
                )
                result["act"] = act_kr
                result["cut_num"] = cut_num
                result["desc"] = scene_info.get("text", "")
                result["scene_info"] = scene_info
                st.session_state.storyboard_data.append(result)

                scene_counter += 1
                progress_bar.progress(scene_counter / total_scenes)

        st.rerun()

    # ── 생성 완료 — 결과 표시 ──────────────────────────────────
    else:
        plan = st.session_state.act_plan
        st.subheader(f"🎞️ [{name}]의 완성 스토리보드")
        if plan:
            st.info(
                f"🎬 장르: **{plan.get('genre')}** | 속도감: **{plan.get('pacing')}** | "
                f"총 {len(st.session_state.storyboard_data)}컷 | "
                f"평균 씬 길이: {plan.get('avg_scene_duration','N/A')}초"
            )

        ACT_CONFIG = [("1막","설정","🌅"), ("2막","대립","⚡"), ("3막","해결","🌟")]

        for act_kr, act_desc_kr, emoji in ACT_CONFIG:
            act_scenes = [s for s in st.session_state.storyboard_data if s.get("act") == act_kr]
            if not act_scenes:
                continue

            act_dur = sum(s.get("scene_info",{}).get("duration",0) for s in act_scenes)
            st.markdown(f"#### {emoji} {act_kr} — {act_desc_kr} (약 {act_dur}초)")
            cols = st.columns(len(act_scenes))

            for col_i, scene in enumerate(act_scenes):
                with cols[col_i]:
                    scores = scene.get("best_scores", {})
                    scene_type = scene.get("scene_type", "face_visible")
                    cfg = SCENE_TYPE_CONFIG.get(scene_type, {})
                    si = scene.get("scene_info", {})

                    if scene.get("image_bytes") is None:
                        st.error(f"❌ {act_kr} {scene['cut_num']}컷 이미지 생성 실패")
                    else:
                        st.image(scene["image_bytes"],
                                 caption=f"{act_kr} {scene['cut_num']}컷")
                    st.caption(f"{cfg.get('emoji','')} {cfg.get('label', scene_type)} | {si.get('camera','')} | {si.get('duration',0)}초")
                    st.caption(score_badge(scores.get("deepface",-1), "DeepFace", df_thr, cfg.get("use_deepface",True)))
                    st.caption(score_badge(scores.get("clip",-1), "CLIP image", clip_thr, cfg.get("use_clip",True)))
                    st.caption(score_badge(scores.get("clip_text",-1), "CLIP text", clip_text_thr, cfg.get("use_clip_text",False)))

                    st.caption("✅ 통과" if scene.get("passed") else ("❌ 생성 실패" if scene.get("failed") else "⚠️ 최고점 채택"))
                    st.caption(scene['desc'][:45] + "...")

                    failed = [a for a in scene.get("all_attempts",[]) if not a["passed"]]
                    if failed:
                        with st.expander(f"🔬 탈락 {len(failed)}장"):
                            st.image(ref_bytes, width=140, caption="레퍼런스")
                            st.divider()
                            for a in scene.get("all_attempts",[]):
                                ch = a["check"]
                                sc_tag = f" 🔧{a.get('strategy','')}" if a.get("self_corrected") else ""
                                st.image(a["image_bytes"],
                                         caption=f"{a['attempt']}차{sc_tag} | {'🟢통과' if a['passed'] else '🔴탈락'}",
                                         use_container_width=True)
                                st.caption(
                                    f"DeepFace:{ch.get('deepface_score',-1)} | "
                                    f"CLIP:{ch.get('clip_score',-1)} | "
                                    f"CLIP-Text:{ch.get('clip_text_score',-1)}"
                                )
                                if a.get("fail_reason"):
                                    st.caption(f"탈락: {a['fail_reason']}")
                                st.divider()

            st.divider()

        # ── 하단 저장 버튼 ──────────────────────────────────────
        st.markdown("### 💾 저장")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            zip_data = make_image_zip(st.session_state.storyboard_data)
            st.download_button(
                label="🖼️ 이미지 전체 저장 (ZIP)",
                data=zip_data,
                file_name=f"{name}_storyboard.zip",
                mime="application/zip",
                use_container_width=True
            )
        with col2:
            txt_data = make_scenario_txt(
                name,
                st.session_state.act_plan,
                st.session_state.scenes_by_act,
                st.session_state.storyboard_data
            )
            st.download_button(
                label="📄 시나리오 저장 (TXT)",
                data=txt_data.encode("utf-8"),
                file_name=f"{name}_scenario.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col3:
            if st.button("🔧 샘플 설정으로 돌아가기", use_container_width=True):
                st.session_state.stage = 'sample'
                st.session_state.storyboard_data = []
                st.rerun()
        with col4:
            if st.button("🔄 처음부터 다시", use_container_width=True):
                for key in defaults:
                    st.session_state[key] = defaults[key]
                st.rerun()