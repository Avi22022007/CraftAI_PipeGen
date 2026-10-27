# Workplace Safety Sentinel (PipeGen Track A)
Team: Elite Innovators (ad824d56)

Detects and tracks people in a video and raises an alert when someone enters a marked hazard zone. Alerts are published over MQTT and logged as JSON.

## Pipeline (built with CraftifAI PipeGen v0.5.1, all five stages)
- Requirement -> Pipeline IR -> Model -> Compiler -> Run
- Input: video file (AI-generated 10 s clip, 2116x1080; disclosed)
- Tasks: person detection, person tracking, hazard-zone entry alert
- Model: YOLO11s (COCO) compiled to a TensorRT 10.3.0 FP32 engine
- Tracker: NvSORT (generated pipeline)
- Outputs: MQTT topic hazard/alerts (localhost:1883), output/metadata.json, live display with zone overlay
- Zone polygon (px): (528,792), (752,426), (1673,480), (1707,885)

## Results (from PipeGen reports; RTX 3050 4 GB, WSL2)
- Engine: 33.72 ms/frame (~30 fps), 41.7 W, build 12.8 s, 49.2 MB, 234 layers
- Live run: 301 frames in 14.1 s (~21 fps end to end, display-limited), 301 MQTT messages
- Alert: one entry event at 3.53 s; person inside the zone for frames 104-195 (3.53-6.57 s); entry_total = 1
- Accuracy: mAP@0.5 0.4782 on a 30-sample COCO slice, identical to the ONNX (generic benchmark, not this video)
- Note: the detector also reports background objects (bench, tv); the alert logic uses only tracked persons.
- Not claimed: INT8, FirmGen, Closed Loop, Jetson.

## Run
bash run.sh (NVIDIA GPU + Docker; run mosquitto for MQTT)

## Evidence
See evidence/: Pipeline IR graph, engine report, speed report, run summary, accuracy report, alert_summary.txt, sample metadata. Run recording:https://drive.google.com/file/d/1g5TGoPilv3JXZ8OxGbR4JM5lg4eTWNtp/view?usp=drivesdk
E     
     
