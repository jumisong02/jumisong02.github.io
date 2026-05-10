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

# ── 라이브러리 임포트 ──────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image as PILImage
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import torch
    from transformers import AutoProcessor, AutoModel
    from PIL import Image as PILImage
    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False

# ── API 클라이언트 ─────────────────────────────────────────────
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ── 세션 초기화 ────────────────────────────────────────────────
defaults = {
    'stage': 'input',           # input → character → scenario → sample → storyboard
    'character_image': None,
    'storyboard_data': [],
    'sample_data': [],
    'topic': '',
    'char_name': '',
    'char_description': '',
    'act_plan': None,
    'scenes_by_act': None,      # 확정된 씬 텍스트 (사용자 언어)
    'check_method': 'A: DeepFace + CLIP',
    'df_threshold': 60,
    'clip_threshold': 60,
    'dino_threshold': 60,
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
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

@st.cache_resource
def load_dino_model():
    from transformers import AutoProcessor, AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    return model, processor

def bytes_to_pil(img_bytes):
    from PIL import Image as PILImage
    return PILImage.open(io.BytesIO(img_bytes)).convert("RGB")


# ══════════════════════════════════════════════════════════════
# [헬퍼] 검수 함수들
# ══════════════════════════════════════════════════════════════
def check_deepface_clip(ref_bytes, gen_bytes):
    result = {"deepface_score": -1, "deepface_passed": True,
              "deepface_reason": "미실행", "clip_score": -1, "clip_reason": "미실행"}
    ref_path = gen_path = None
    if DEEPFACE_AVAILABLE:
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
            result["deepface_reason"] = f"유사도 {score}점 (distance={dist:.3f})"
        except Exception as e:
            err = str(e)
            result["deepface_reason"] = "얼굴 미검출" if "face" in err.lower() else f"오류: {err}"
        finally:
            for p in [ref_path, gen_path]:
                try: os.unlink(p) if p else None
                except: pass
    if CLIP_AVAILABLE:
        try:
            clip_model, clip_proc = load_clip_model()
            inputs = clip_proc(images=[bytes_to_pil(ref_bytes), bytes_to_pil(gen_bytes)],
                               return_tensors="pt", padding=True)
            with torch.no_grad():
                feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sim = (feats[0] @ feats[1]).item()
            clip_score = round(max(0.0, sim) * 100, 1)
            result["clip_score"] = clip_score
            result["clip_reason"] = f"CLIP {clip_score}점 (cosine={sim:.3f})"
        except Exception as e:
            result["clip_reason"] = f"CLIP 오류: {e}"
    return result

def check_dino(ref_bytes, gen_bytes):
    result = {"dino_score": -1, "dino_reason": "미실행"}
    if not DINO_AVAILABLE:
        result["dino_reason"] = "DINOv2 미설치"
        return result
    try:
        dino_model, dino_proc = load_dino_model()
        inputs = dino_proc(images=[bytes_to_pil(ref_bytes), bytes_to_pil(gen_bytes)],
                           return_tensors="pt")
        with torch.no_grad():
            outputs = dino_model(**inputs)
        feats = outputs.last_hidden_state[:, 0, :]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats[0] @ feats[1]).item()
        dino_score = round(max(0.0, sim) * 100, 1)
        result["dino_score"] = dino_score
        result["dino_reason"] = f"DINOv2 {dino_score}점 (cosine={sim:.3f})"
    except Exception as e:
        result["dino_reason"] = f"DINOv2 오류: {e}"
    return result

def run_similarity_check(ref_bytes, gen_bytes, check_method, df_thr, clip_thr, dino_thr):
    result_a = check_deepface_clip(ref_bytes, gen_bytes)
    result_b = check_dino(ref_bytes, gen_bytes)

    if check_method.startswith("A"):
        df_s = result_a["deepface_score"]
        cl_s = result_a["clip_score"]
        df_ok = result_a["deepface_passed"] and (df_s == -1 or df_s >= df_thr)
        cl_ok = (cl_s == -1 or cl_s >= clip_thr)
        overall = df_ok and cl_ok
        if df_s == -1: fail = None
        elif not result_a["deepface_passed"]: fail = f"DeepFace 기준 미달 ({df_s}점)"
        elif not df_ok: fail = f"DeepFace 슬라이더 미달 ({df_s}점 < {df_thr}점)"
        elif not cl_ok: fail = f"CLIP 기준 미달 ({cl_s}점 < {clip_thr}점)"
        else: fail = None
    else:
        di_s = result_b["dino_score"]
        overall = (di_s == -1 or di_s >= dino_thr)
        fail = None if (di_s == -1 or overall) else f"DINOv2 기준 미달 ({di_s}점 < {dino_thr}점)"

    return {**result_a, **result_b, "overall_passed": overall, "fail_reason": fail}


# ══════════════════════════════════════════════════════════════
# [헬퍼] 이미지 생성
# ══════════════════════════════════════════════════════════════
def build_prompt(name, desc, scene_text, prev_score=-1, attempt=1):
    if attempt == 1 or prev_score < 0:
        return (
            f"You are a cinematic image generator. Character in reference image: {name}. "
            f"Generate hyperrealistic cinematic film still (16:9) where {name}: {scene_text}. "
            f"CRITICAL: Keep {name}'s face, body, costume EXACTLY identical to reference. "
            "ARRI Alexa 65, anamorphic lens, 8K, dramatic lighting, shallow DOF."
        )
    return (
        f"Self-correction attempt #{attempt}. Previous similarity: {prev_score:.1f}/100 (FAILED). "
        f"Show {name}'s face MORE clearly. Closer framing. Prioritize face/costume accuracy. "
        f"Character {name} ({desc}): {scene_text}. 16:9 cinematic. Face must match reference exactly."
    )

def generate_image(name, desc, scene_text, ref_bytes, prev_score=-1, attempt=1):
    prompt = build_prompt(name, desc, scene_text, prev_score, attempt)
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
# [헬퍼] 씬 1개 생성 + 검수 루프
# ══════════════════════════════════════════════════════════════
def generate_and_check(name, desc, scene_text, ref_bytes,
                       check_method, df_thr, clip_thr, dino_thr, max_retries,
                       label, status_box):
    best_image = None
    best_scores = {}
    passed = False
    all_attempts = []
    prev_score = -1

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            status_box.info(f"🎨 **{label}** — 생성 중...")
        else:
            status_box.info(f"🔄 **{label}** — {attempt}차 Self-Correction (이전: {prev_score:.1f}점)")

        try:
            image_bytes = generate_image(name, desc, scene_text, ref_bytes, prev_score, attempt)
        except Exception as e:
            status_box.error(f"생성 오류: {e}")
            break

        if not image_bytes:
            status_box.warning("이미지 응답 없음, 재시도...")
            continue

        check = run_similarity_check(ref_bytes, image_bytes, check_method, df_thr, clip_thr, dino_thr)
        attempt_passed = check["overall_passed"]
        fail_reason = check.get("fail_reason")

        active_score = check.get("deepface_score", -1) if check_method.startswith("A") else check.get("dino_score", -1)
        if active_score != -1:
            prev_score = active_score

        if best_image is None or active_score > best_scores.get("active", -1):
            best_image = image_bytes
            best_scores = {
                "active": active_score,
                "deepface": check.get("deepface_score", -1),
                "clip": check.get("clip_score", -1),
                "dino": check.get("dino_score", -1),
            }

        all_attempts.append({
            "attempt": attempt, "image_bytes": image_bytes,
            "check": check, "passed": attempt_passed,
            "fail_reason": fail_reason, "self_corrected": attempt > 1,
        })

        if attempt_passed:
            passed = True
            status_box.success(f"✅ **{label}** 통과!")
            break
        else:
            status_box.warning(f"⚠️ **{label}** {attempt}차 탈락 — {fail_reason}")

    if not passed and best_image:
        status_box.warning(
            f"⚠️ **{label}** 최고점 채택 "
            f"(DeepFace:{best_scores.get('deepface',-1)} / "
            f"CLIP:{best_scores.get('clip',-1)} / "
            f"DINOv2:{best_scores.get('dino',-1)})"
        )

    return {"image_bytes": best_image, "passed": passed,
            "best_scores": best_scores, "all_attempts": all_attempts}


# ══════════════════════════════════════════════════════════════
# [헬퍼] 점수 배지
# ══════════════════════════════════════════════════════════════
def score_badge(score, label, threshold=60):
    if score == -1: return f"🔵 {label}: N/A"
    color = "🟢" if score >= threshold else "🔴"
    return f"{color} {label}: {score}점"


# ══════════════════════════════════════════════════════════════
# [헬퍼] 스토리보드 ZIP 다운로드 데이터 생성
# ══════════════════════════════════════════════════════════════
def make_zip(storyboard_data):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in storyboard_data:
            act = scene.get("act", "unknown")
            cut = scene.get("cut_num", 0)
            filename = f"{act}_{cut:02d}컷.png"
            zf.writestr(filename, scene["image_bytes"])
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# [단계 0] 캐릭터 설정
# ══════════════════════════════════════════════════════════════
if st.session_state.stage == 'input':
    st.subheader("🧑‍🎨 주인공 외형 설정")

    avail = []
    avail.append("✅ DeepFace" if DEEPFACE_AVAILABLE else "❌ DeepFace (pip install deepface tf-keras)")
    avail.append("✅ CLIP" if CLIP_AVAILABLE else "❌ CLIP (pip install transformers torch)")
    avail.append("✅ DINOv2" if DINO_AVAILABLE else "❌ DINOv2 (pip install transformers torch)")
    st.info("검수 모델: " + " | ".join(avail))

    topic = st.text_input("주인공 외형/연출 의도",
                          value=st.session_state.topic,
                          placeholder="예: 플라스틱 질감의 마네킹 같은 얼굴을 한 사이버펑크 캐릭터")
    char_name = st.text_input("캐릭터 이름 (영문 권장)",
                              value=st.session_state.char_name,
                              placeholder="예: NOVA, MIRA, ZERO ...")

    _, btn_col = st.columns([3, 1])
    with btn_col:
        clicked_generate_char = st.button("캐릭터 3면도 생성하기", use_container_width=True)
    if clicked_generate_char:
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
                    result = client.models.generate_images(
                        model='imagen-4.0-generate-001',
                        prompt=(f"Photorealistic 3-view turnaround of {name}: {desc}. "
                                "Front/side/back on white background. 8K, studio lighting."),
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
    desc = st.session_state.char_description
    st.subheader(f"🧑‍🎨 [{name}] 3면도 확인")
    st.image(st.session_state.character_image, caption=name)
    st.caption(f"외형 키워드: {desc}")
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

    _, btn_col = st.columns([3, 1])
    with btn_col:
        clicked_gen_scenario = st.button("🎬 시나리오 자동 생성", use_container_width=True)
    if clicked_gen_scenario:
        if not plot_input.strip():
            st.warning("줄거리를 입력해 주세요.")
        else:
            # 1단계: 장르/속도감 → 씬 수 결정
            with st.spinner("장르 및 속도감 분석 중..."):
                try:
                    plan_resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"""
Analyze this 5-minute film plot. Decide scene count per act (min 2, max 5).
Return ONLY JSON:
{{"genre":"...","pacing":"fast/medium/slow","reasoning":"한국어로 간단히","act1_scenes":N,"act2_scenes":N,"act3_scenes":N}}
Plot: {plot_input}
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
                    act1_n, act2_n, act3_n = 3, 3, 3
                    st.session_state.act_plan = None

            # 2단계: 시나리오 생성 (사용자 언어 — 한국어)
            with st.spinner("시나리오 작성 중..."):
                try:
                    scene_resp = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"""
당신은 단편 영화 시나리오 작가입니다.
주인공 이름: {name}, 외형: {desc}

아래 줄거리를 3막 구조로 확장하고 각 막별 씬 묘사를 작성해주세요.
- 1막(설정): {act1_n}개 씬
- 2막(대립): {act2_n}개 씬
- 3막(해결): {act3_n}개 씬

규칙:
- 각 씬은 반드시 {name}을 언급해야 합니다
- 한국어로 작성 (카메라 앵글, 조명, 액션 포함)
- 반드시 아래 형식으로만 출력:
1막_씬1: [묘사]
1막_씬2: [묘사]
...
2막_씬1: [묘사]
...
3막_씬1: [묘사]
...

줄거리: {plot_input}
"""
                    )
                    raw_lines = scene_resp.text.strip().split('\n')
                    scenes_by_act = {"1막": [], "2막": [], "3막": []}
                    act_map = {"1막": "1막", "2막": "2막", "3막": "3막"}

                    for line in raw_lines:
                        line = line.strip()
                        for prefix, act_kr in act_map.items():
                            if line.startswith(prefix):
                                scenes_by_act[act_kr].append(line)
                                break

                    st.session_state.scenes_by_act = scenes_by_act
                    st.rerun()

                except Exception as e:
                    st.error(f"시나리오 생성 오류: {e}")

    # 생성된 시나리오가 있으면 표시 + 수정
    if st.session_state.scenes_by_act:
        plan = st.session_state.act_plan
        if plan:
            st.success(
                f"장르: **{plan.get('genre')}** | 속도감: **{plan.get('pacing')}** | "
                f"총 {sum(len(v) for v in st.session_state.scenes_by_act.values())}컷"
            )
            st.caption(f"결정 근거: {plan.get('reasoning')}")

        st.markdown("### 📖 생성된 시나리오 — 직접 수정 가능합니다")
        st.info("💡 각 씬의 내용을 자유롭게 수정한 뒤 '이 시나리오로 확정' 버튼을 누르세요.")

        ACT_EMOJI = {"1막": "🌅", "2막": "⚡", "3막": "🌟"}
        ACT_DESC = {"1막": "설정", "2막": "대립", "3막": "해결"}

        edited_scenes = {"1막": [], "2막": [], "3막": []}

        for act_kr in ["1막", "2막", "3막"]:
            scenes_list = st.session_state.scenes_by_act.get(act_kr, [])
            if not scenes_list:
                continue
            st.markdown(f"#### {ACT_EMOJI[act_kr]} {act_kr} — {ACT_DESC[act_kr]}")
            for j, scene_text in enumerate(scenes_list):
                edited = st.text_area(
                    f"{act_kr} {j+1}번째 씬",
                    value=scene_text,
                    key=f"scene_{act_kr}_{j}",
                    height=80
                )
                edited_scenes[act_kr].append(edited)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 시나리오 다시 생성", use_container_width=True):
                st.session_state.scenes_by_act = None
                st.rerun()
        with col2:
            if st.button("✅ 이 시나리오로 확정 → 샘플 테스트", use_container_width=True):
                st.session_state.scenes_by_act = edited_scenes
                st.session_state.stage = 'sample'
                st.rerun()


# ══════════════════════════════════════════════════════════════
# [단계 3] 샘플 테스트 (각 막 1컷씩 = 3컷)
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'sample':
    name = st.session_state.char_name
    desc = st.session_state.char_description
    scenes_by_act = st.session_state.scenes_by_act
    ref_bytes = st.session_state.character_image

    st.subheader("🧪 샘플 테스트 (각 막 첫 번째 씬)")
    st.info("전체 생성 전에 각 막의 첫 번째 씬만 먼저 생성해서 검수 설정을 확인합니다.")

    # 검수 설정
    with st.expander("⚙️ 검수 설정", expanded=True):
        check_method = st.radio(
            "재생성 판정 기준",
            ["A: DeepFace + CLIP", "B: DINOv2"],
            index=0 if st.session_state.check_method.startswith("A") else 1,
            horizontal=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            df_thr = st.slider("DeepFace 기준점", 30, 90,
                               st.session_state.df_threshold, 5)
        with col2:
            clip_thr = st.slider("CLIP 기준점", 30, 90,
                                 st.session_state.clip_threshold, 5)
        with col3:
            dino_thr = st.slider("DINOv2 기준점", 30, 90,
                                 st.session_state.dino_threshold, 5)
        max_retries = st.slider("최대 재시도", 1, 5, st.session_state.max_retries)

        # 설정 세션 저장
        st.session_state.check_method = check_method
        st.session_state.df_threshold = df_thr
        st.session_state.clip_threshold = clip_thr
        st.session_state.dino_threshold = dino_thr
        st.session_state.max_retries = max_retries

    _, btn_col = st.columns([3, 1])
    with btn_col:
        clicked_gen_sample = st.button("🧪 샘플 3컷 생성", use_container_width=True)
    if clicked_gen_sample:
        sample_data = []
        progress = st.progress(0)

        for act_i, act_kr in enumerate(["1막", "2막", "3막"]):
            scenes_list = scenes_by_act.get(act_kr, [])
            if not scenes_list:
                continue

            # 각 막의 첫 번째 씬만 사용
            sample_scene = scenes_list[0]
            label = f"{act_kr} 샘플"
            status_box = st.empty()

            result = generate_and_check(
                name, desc, sample_scene, ref_bytes,
                check_method, df_thr, clip_thr, dino_thr, max_retries,
                label, status_box
            )
            result["act"] = act_kr
            result["cut_num"] = 1
            result["desc"] = sample_scene
            sample_data.append(result)
            progress.progress((act_i + 1) / 3)

        st.session_state.sample_data = sample_data
        st.rerun()

    # 샘플 결과 표시
    if st.session_state.sample_data:
        st.markdown("### 🖼️ 샘플 결과")
        cols = st.columns(3)
        for i, sample in enumerate(st.session_state.sample_data):
            with cols[i]:
                act_kr = sample.get("act", f"{i+1}막")
                scores = sample.get("best_scores", {})
                st.image(sample["image_bytes"], caption=f"{act_kr} 샘플")
                st.caption(score_badge(scores.get("deepface", -1), "DeepFace", df_thr))
                st.caption(score_badge(scores.get("clip", -1), "CLIP", clip_thr))
                st.caption(score_badge(scores.get("dino", -1), "DINOv2", dino_thr))
                st.caption("✅ 통과" if sample.get("passed") else "⚠️ 최고점 채택")

                failed = [a for a in sample.get("all_attempts", []) if not a["passed"]]
                if failed:
                    with st.expander(f"🔬 탈락 {len(failed)}장"):
                        st.image(ref_bytes, width=150, caption="레퍼런스")
                        for a in sample.get("all_attempts", []):
                            ch = a["check"]
                            sc_tag = " 🔧SC" if a.get("self_corrected") else ""
                            status = "🟢 통과" if a["passed"] else "🔴 탈락"
                            st.image(a["image_bytes"],
                                     caption=f"{a['attempt']}차{sc_tag} | {status}",
                                     use_container_width=True)
                            st.caption(
                                f"DeepFace: {ch.get('deepface_score',-1)}점 | "
                                f"CLIP: {ch.get('clip_score',-1)}점 | "
                                f"DINOv2: {ch.get('dino_score',-1)}점"
                            )
                            if a.get("fail_reason"):
                                st.caption(f"탈락: {a['fail_reason']}")

        st.divider()
        st.markdown("**샘플이 만족스러우신가요?**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📝 시나리오부터 다시", use_container_width=True):
                st.session_state.stage = 'scenario'
                st.session_state.sample_data = []
                st.rerun()
        with col2:
            if st.button("🔧 설정 바꾸고 샘플 재생성", use_container_width=True):
                st.session_state.sample_data = []
                st.rerun()
        with col3:
            if st.button("✅ 전체 스토리보드 생성", use_container_width=True):
                st.session_state.stage = 'storyboard'
                st.session_state.storyboard_data = []
                st.rerun()


# ══════════════════════════════════════════════════════════════
# [단계 4] 전체 스토리보드 생성
# ══════════════════════════════════════════════════════════════
elif st.session_state.stage == 'storyboard':
    name = st.session_state.char_name
    desc = st.session_state.char_description
    scenes_by_act = st.session_state.scenes_by_act
    ref_bytes = st.session_state.character_image

    check_method = st.session_state.check_method
    df_thr = st.session_state.df_threshold
    clip_thr = st.session_state.clip_threshold
    dino_thr = st.session_state.dino_threshold
    max_retries = st.session_state.max_retries

    # 아직 생성 안 됐으면 생성 시작
    if not st.session_state.storyboard_data:
        st.subheader(f"🎬 [{name}] 전체 스토리보드 생성 중...")
        st.caption(f"검수 방식: {check_method} | 재시도: {max_retries}회")

        total_scenes = sum(len(v) for v in scenes_by_act.values())
        progress_bar = st.progress(0)
        scene_counter = 0

        for act_kr in ["1막", "2막", "3막"]:
            scenes_list = scenes_by_act.get(act_kr, [])
            for cut_idx, scene_text in enumerate(scenes_list):
                cut_num = cut_idx + 1
                label = f"{act_kr} {cut_num}컷"
                status_box = st.empty()

                result = generate_and_check(
                    name, desc, scene_text, ref_bytes,
                    check_method, df_thr, clip_thr, dino_thr, max_retries,
                    label, status_box
                )
                result["act"] = act_kr
                result["cut_num"] = cut_num
                result["desc"] = scene_text
                st.session_state.storyboard_data.append(result)

                scene_counter += 1
                progress_bar.progress(scene_counter / total_scenes)

        st.rerun()

    # ── 생성 완료 — 결과 표시 ──────────────────────────────────
    else:
        st.subheader(f"🎞️ [{name}]의 완성 스토리보드")

        plan = st.session_state.act_plan
        if plan:
            st.info(
                f"🎬 장르: **{plan.get('genre')}** | 속도감: **{plan.get('pacing')}** | "
                f"총 {len(st.session_state.storyboard_data)}컷"
            )

        ACT_CONFIG = [
            ("1막", "설정", "🌅"), ("2막", "대립", "⚡"), ("3막", "해결", "🌟")
        ]

        for act_kr, act_desc_kr, emoji in ACT_CONFIG:
            act_scenes = [s for s in st.session_state.storyboard_data if s.get("act") == act_kr]
            if not act_scenes:
                continue

            st.markdown(f"#### {emoji} {act_kr} — {act_desc_kr}")
            cols = st.columns(len(act_scenes))

            for col_i, scene in enumerate(act_scenes):
                with cols[col_i]:
                    scores = scene.get("best_scores", {})
                    st.image(scene["image_bytes"], caption=f"{act_kr} {scene['cut_num']}컷")
                    st.caption(score_badge(scores.get("deepface", -1), "DeepFace", df_thr))
                    st.caption(score_badge(scores.get("clip", -1), "CLIP", clip_thr))
                    st.caption(score_badge(scores.get("dino", -1), "DINOv2", dino_thr))
                    st.caption("✅ 통과" if scene.get("passed") else "⚠️ 최고점 채택")
                    st.caption(scene['desc'][:50] + "...")

                    failed = [a for a in scene.get("all_attempts", []) if not a["passed"]]
                    if failed:
                        with st.expander(f"🔬 탈락 {len(failed)}장"):
                            st.image(ref_bytes, width=150, caption="레퍼런스")
                            st.divider()
                            for a in scene.get("all_attempts", []):
                                ch = a["check"]
                                sc_tag = " 🔧SC" if a.get("self_corrected") else ""
                                status = "🟢 통과" if a["passed"] else "🔴 탈락"
                                st.image(a["image_bytes"],
                                         caption=f"{a['attempt']}차{sc_tag} | {status}",
                                         use_container_width=True)
                                st.caption(
                                    f"DeepFace: {ch.get('deepface_score',-1)}점 | "
                                    f"CLIP: {ch.get('clip_score',-1)}점 | "
                                    f"DINOv2: {ch.get('dino_score',-1)}점"
                                )
                                if a.get("fail_reason"):
                                    st.caption(f"탈락: {a['fail_reason']}")
                                st.divider()

            st.divider()

        # ── 하단 버튼 ──────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            # 전체 이미지 ZIP 다운로드
            zip_data = make_zip(st.session_state.storyboard_data)
            st.download_button(
                label="💾 전체 이미지 저장 (ZIP)",
                data=zip_data,
                file_name=f"{name}_storyboard.zip",
                mime="application/zip",
                use_container_width=True
            )
        with col2:
            if st.button("🔧 샘플 설정으로 돌아가기", use_container_width=True):
                st.session_state.stage = 'sample'
                st.session_state.storyboard_data = []
                st.rerun()
        with col3:
            if st.button("🔄 처음부터 다시", use_container_width=True):
                for key in ['stage', 'character_image', 'storyboard_data', 'sample_data',
                            'topic', 'char_name', 'char_description', 'act_plan', 'scenes_by_act']:
                    st.session_state[key] = defaults[key]
                st.rerun()