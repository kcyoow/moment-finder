# pipeline — 6장의 흐름과 로컬 웹사이트

[6장](../experiments/06_pipeline_site/)에서 만든 것이다. 판정 모델(H2O-Lightning-4B)은 Modal의 GPU에서 돌고, 질문 쓰기와 다시 확인은 Gemini 4 Hydrogen이 한다. 공개 Space([`../space/`](../space/))는 이 흐름을 공개 모델 둘로 옮긴 것이다.

## 돌리는 법

[uv](https://docs.astral.sh/uv/), Modal 계정과 `modal` 명령, Vertex AI를 쓸 수 있는 `gcloud` 로그인이 있어야 한다. `<modal-profile>`은 자기 Modal 프로필 이름, `<gcp-project>`는 자기 Google Cloud 프로젝트로 바꾼다.

```bash
cd pipeline && MODAL_PROFILE=<modal-profile> modal deploy mf_serve_modal.py && cd ..
```

```bash
MODAL_PROFILE=<modal-profile> uv run --no-project --with fastapi --with uvicorn --with python-multipart --with numpy --with pillow --with imageio-ffmpeg --with modal==1.5.5 python pipeline/web/server.py
```

웹사이트는 http://127.0.0.1:8770 에 뜬다.

| 앞에 붙이는 것 | 하는 일 |
|---|---|
| `MF_VERIFY=0` | 다시 확인 단계를 건너뛴다 |
| `MF_VERIFY=h2o` | 다시 확인을 Gemini 대신 판정 모델이 스스로 한다([9장](../experiments/09_self_recheck/)). `mf_serve2_modal.py`를 올려 둔 것을 쓴다 |

판정 모델의 GPU는 켜져 있는 동안 초당 약 $0.001(시간당 약 $3.7)이고, 5분 동안 쓰지 않으면 스스로 꺼진다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `mf.py` | 프레임 뽑기, 요청 묶음 만들기, 클립 자르기 |
| `mf_core.py` | 질문들의 답을 점수 하나로 합치기, 점수를 클립으로 묶는 규칙, 후보를 목록 여럿에서 돌아가며 뽑기 |
| `mf_writer.py`, `writer_brief.md` | 요청을 질문 3~5개로 바꾸기 (LLM은 영상을 보지 않는다) |
| `mf_modal.py`, `mf_serve_modal.py`, `mf_serve2_modal.py` | Modal에서 도는 판정 모델. 앞의 것은 한 번 돌고 꺼지는 작업, 뒤의 둘은 올려 두는 앱 |
| `mf_verify.py`, `verifier_brief.md` | 후보의 프레임 8장을 Gemini가 다시 확인 |
| `mf_recheck.py`, `verify_question_brief.md` | 사이트의 다시 확인 단계. 판정 모델이 스스로 확인하는 방식 포함 |
| `web/server.py`, `web/index.html` | 웹사이트 |
| `mf_*_check.py`, `mf_*_ceiling.py` | 받아 둔 답으로 한 확인들(합치는 식 견주기, 다시 확인의 상한 계산, 정답 표시 쪼개 보기 등). 자료 폴더가 없어 그대로는 돌지 않는다 |
