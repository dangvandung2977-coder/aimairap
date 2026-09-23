# PvP 2-bot qua mineflayer (Minecraft 1.21.1)

Hai bot mineflayer đánh nhau, mỗi con do checkpoint PPO điều khiển qua đúng
protocol localhost của bản mod (`deployment/protocol.py`, version 1).
Python không cần sửa gì.

## Chuẩn bị

1. Node 18+ (`node --version`), rồi: `npm install`
2. Server Minecraft **1.21.1** bất kỳ (vanilla/Paper), `online-mode=false`.
   Gợi ý: superflat, `/gamerule keepInventory true`, difficulty Normal.
3. Chạy server Python (repo root): `python -m deployment.inference_server`
4. Cho 2 bot kit giống lúc train (thay tên cho đúng):
   `/give PvPBot_A iron_sword 1`, `/give PvPBot_A shield 2`,
   `/give PvPBot_A bread 5`, `/give PvPBot_A golden_apple 1`,
   `/give PvPBot_A ender_pearl 4`, `/give PvPBot_A cobweb 4`,
   `/give PvPBot_A dirt 16`, `/give PvPBot_A bow 1`, `/give PvPBot_A arrow 16`
   (lặp lại cho `PvPBot_B`; bot tự cầm kiếm + khiên offhand khi spawn)
5. Chạy: `node bot2v2.js --host 127.0.0.1 --port 25565`

Bot quyết định mỗi 4 tick (= `frame_skip` lúc train). Mất kết nối bridge →
đứng yên giữ camera, tự nối lại.

## Test không cần Minecraft

`npm test` (= `node test_encoder.js`): check 73 chiều, công thức bearing/
walls/counts, decoder hợp lệ/bất hợp lệ.

Dump obs để golden replay: `node bot2v2.js --dump-dir ./dumps`, rồi
`python -m deployment.replay_golden dumps/` (chạy ở repo root).

## Giới hạn đã biết (trung thực)

- `absorption` luôn báo 0 (mineflayer không expose).
- Trạng thái khiên/ăn/effect của địch lấy trực tiếp từ object bot kia
  (chỉ đúng vì 2 bot cùng process — đánh với người thật thì không có).
- Khiên ở offhand cố định; policy chọn slot 1 được giữ nguyên vị trí hotbar.
- cobweb/block/strength chọn slot nhưng chưa đặt/uống (phase sau, như bản mod).
- Sát thương đo bằng health-delta quan sát được (sát thương thật, mọi nguồn).
- Không tường bao: dùng arena ảo 64 block quanh điểm spawn, sensor pillar rỗng.
