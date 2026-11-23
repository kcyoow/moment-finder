"""마인크래프트 행동 기록에서 사건의 시각을 뽑는다. 표준 라이브러리만 쓴다. (하이라이트 실험의 정답)

VPT 기록(jsonl)의 stats는 게임의 누적 통계다. 어떤 값이 늘어난 틱이 그 사건이 일어난 틱이다(i번째 줄 = 영상의 i번째 프레임, 초당 20틱).

뽑는 사건
- fight       : 준 피해 (minecraft.custom:minecraft.damage_dealt)
- kill        : 동물·몬스터를 잡음 (minecraft.kill_entity:*). fight의 일부와 겹친다
- hurt        : 입은 피해 (minecraft.custom:minecraft.damage_taken)
- crafting    : 제작 (minecraft.craft_item:*). level = 크기(3 다이아몬드 도구, 2 작업대·화로·도구·철 주괴, 1 그 밖)
- ore         : 광석 블록을 캠 (minecraft.mine_block:*_ore)
- find        : 다이아몬드를 주움 (minecraft.pickup:minecraft.diamond)
- tool_break  : 도구가 부서짐 (minecraft.break_item:*)

받은 파일은 믿지 않는 자료이므로 -I로 돌린다.
실행: python3 -I experiments/frame_limit/scripts/highlight/hl_events.py
결과: data/mc_events.json
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # experiments/frame_limit
TICKS = 20
TOOLS = ("_pickaxe", "_axe", "_sword", "_shovel", "_hoe")
STATIONS = {"crafting_table", "furnace", "iron_ingot"}


def craft_level(item):
    if item.startswith("diamond_"):
        return 3
    return 2 if item in STATIONS or item.endswith(TOOLS) else 1


def main():
    videos = json.loads((HERE / "data" / "mc_items.json").read_text())["videos"]
    events = []
    for vid, v in videos.items():
        path = (HERE / v["file"]).with_suffix(".jsonl")
        prev = None
        with open(path) as f:
            for i, line in enumerate(f):
                stats = json.loads(line).get("stats") or {}
                if prev is not None:
                    for key, value in stats.items():
                        if value <= prev.get(key, 0):
                            continue
                        group, _, item = key.partition(":")
                        group, item = group.replace("minecraft.", ""), item.replace("minecraft.", "")
                        ev = {"video": vid, "t": round(i / TICKS, 2), "item": item, "amount": value - prev.get(key, 0)}
                        if group == "craft_item":
                            events.append({**ev, "kind": "crafting", "level": craft_level(item)})
                        elif group == "kill_entity":
                            events.append({**ev, "kind": "kill"})
                        elif group == "break_item":
                            events.append({**ev, "kind": "tool_break"})
                        elif group == "mine_block" and item.endswith("_ore"):
                            events.append({**ev, "kind": "ore"})
                        elif group == "pickup" and item == "diamond":
                            events.append({**ev, "kind": "find"})
                        elif key == "minecraft.custom:minecraft.damage_dealt":
                            events.append({**ev, "kind": "fight"})
                        elif key == "minecraft.custom:minecraft.damage_taken":
                            events.append({**ev, "kind": "hurt"})
                prev = stats
    counts = collections.Counter(e["kind"] for e in events)
    levels = collections.Counter(e["level"] for e in events if e["kind"] == "crafting")
    note = "hl_events.py가 행동 기록의 stats에서 뽑은 사건. t는 초(틱 ÷ 20). crafting의 level은 제작의 크기(1~3)"
    (HERE / "data" / "mc_events.json").write_text(json.dumps({"_note": note, "events": events}, ensure_ascii=False, indent=0) + "\n")
    print(f"사건 {len(events)}건: {dict(counts)}. 제작의 크기별: {dict(sorted(levels.items()))}")
    for kind in ("ore", "find", "hurt"):
        print(f"  {kind}: " + ", ".join(f"{e['video'][:5]}@{e['t']:g}s({e['item']})" for e in events if e["kind"] == kind)[:400])


if __name__ == "__main__":
    main()
