---
title: Moment Finder
emoji: 🎬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 30m
models:
- h2oai/h2o-lightning-4b
- Qwen/Qwen3.8-27B-FP8
---

영상을 올리고 찾을 장면을 편하게 적으면, 그 장면이 담긴 대목을 클립으로 잘라 보여 준다. 찾는 장면이 없으면 없다고 말한다.

```
요청 → 중간 LLM(Qwen3.8-27B-FP8)이 "이렇게 찾을까요?" 하고 되묻는다 (글만 본다. 영상은 보지 않는다)
     → 맞다고 하면 판정 모델(H2O-Lightning-4B)이 1초마다 프레임 4장을 보고 답한다 → 후보 6개까지
     → 판정 모델이 후보마다 프레임 8장을 다시 본다 (예/아니오 + 고르기, 둘 다 0.5 이상이어야 통과)
     → 통과한 것만 클립
```

- **GPU 시간**: 한 번에 2~6분(60초 영상 약 3분). ZeroGPU의 하루 한도는 로그인하지 않으면 2분, 무료 계정 5분이다. 화면의 "미리 돌려 둔 예시"는 GPU를 쓰지 않는다
- **판정 모델**: `h2oai/h2o-lightning-4b` 판 `659fc412`. 그 판의 `h2o_lightning_shim.py`·`serve_config.json`(Apache-2.0)이 만드는 프롬프트를 그대로 쓰고, vLLM 대신 transformers로 질문마다 한 번 계산해 답 자리의 확률을 읽는다
- **중간 LLM**: `Qwen/Qwen3.8-27B-FP8` 판 `c3a462cc`을 FP8 그대로 올린다. 생각 모드는 끄고 검색 도구는 없다. 한 번의 답에 1초마다 물을 질문 3개와 다시 확인용 정의(찾는 장면, 헷갈리는 장면 3~4개. 하나는 "찾는 대상만 빠진 같은 동작")를 함께 쓴다
- **다시 확인**: 후보의 프레임 8장을 4장씩 두 번(가장 확신한 1초의 연속 4장, 구간 전체에서 고르게 4장) 같은 판정 모델에 다시 묻는다. "찾는 장면이 담겼나"(예/아니오)와 "찾는 장면, 헷갈리는 장면들, 어느 것도 아님 가운데 무엇인가"(고르기. 선택지 순서를 바꿔 두 번)를 묻고 둘 다 0.5 이상일 때만 통과시킨다
- **파일**: `app.py`(모델 올리기, GPU 함수 둘), `engine.py`(흐름과 화면), `plan.py`(중간 LLM의 계획과 그 검사), `writer.py`·`guide_for_writer.md`·`writer_brief.md`(질문 쓰기 안내), `mf_core.py`(점수 합치기, 클립 규칙)
- **요청을 나누는 까닭**: ZeroGPU는 요청마다 받은 토큰으로 GPU를 잡는데 그 토큰이 한두 분이면 만료된다. 그래서 GPU 호출을 요청 하나에 한 번만 하고, 화면의 타이머가 단계를 하나씩 이어 부른다. 끊기면 "이어서 하기"로 같은 곳부터 다시 한다
- 실험 기록과 만든 과정은 GitHub 저장소에 있다: https://github.com/kcyoow/moment-finder
