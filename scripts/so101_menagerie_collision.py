"""Fetch and hash-check the pinned public Menagerie jaw collision assets."""
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
COMMIT = '3abca2e302625cec9429c49806121ea2855cef81'
BASE = f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/robotstudio_so101/'
FILES = {
    'so101.xml': '982c02c027ec0c62d3459d0a6eb00dfb7dc13a451992e72ce99addc71f209805',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'README.md': 'e2344e7c14290bed85c7907247aecb8bf6cb4392e3de0d7c45680efe9bb9b5a9',
    'assets/wrist_roll_follower_so101_gripper_part0_v1.stl': '370eb470a6ab48b08febcaeb5274d2bcc91246692532ce12c795fb82977def72',
    'assets/moving_jaw_so101_gripper_part0_v1.stl': '3a92e115d1934d4c45cff808c4eff2ea10361a137a7f89940033a1f8a821b88c',
    'assets/moving_jaw_so101_gripper_part1_v1.stl': 'd10f6c671d26f62988ec95efee9370a0657b5833b3c5f8aad59fd69a87ef2aee',
}
VISUALS = {
    'wrist_roll_follower_so101_v1.stl': '4b17b410a12d64ec39554abc3e8054d8a97384b2dc4a8d95a5ecb2a93670f5f4',
    'moving_jaw_so101_v1.stl': '785a9dded2f474bc1d869e0d3dae398a3dcd9c0c345640040472210d2861fa9d',
}


def main():
    out = ROOT/'assets_local/so101_menagerie_collision'
    rows = []
    for name, expected in FILES.items():
        target = out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = target.read_bytes() if target.exists() else urllib.request.urlopen(BASE+name).read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: expected pinned SHA256 {expected}, got {actual}')
        if not target.exists():
            target.write_bytes(data)
        rows.append(dict(name=name, source=BASE+name, sha256=actual, bytes=len(data)))
    for name, expected in VISUALS.items():
        actual = hashlib.sha256((ROOT/'assets_local/so101/assets'/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Current {name} differs from the pinned collision model visual')
    report = dict(commit=COMMIT, files=rows, current_visual_sha256=VISUALS,
                  scope='jaw collision assets only; no source sites, camera, kinematics or controller settings')
    (out/'verified.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
