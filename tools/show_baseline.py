import json

s = json.load(open("logs/suite_phase25.json"))["profiles"]
for k, m in s.items():
    print(k, "win", m["win_rate"], "dealt", round(m["damage_dealt"], 1),
          "taken", round(m["damage_taken"], 1), "hit", round(m["hit_rate"], 2),
          "dist", round(m["mean_dist"], 2), "hp", round(m["hp_left"], 1))
b = json.load(open("logs/phase25_behavior.json"))
print("replay behavior (ep0 per profile): jump strafe air dist")
for r in b:
    print(r[0], r[1], r[2], r[4], r[5])
