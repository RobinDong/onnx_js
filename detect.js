const ort = require("onnxruntime-node");
const inkjet = require("inkjet");
const fs = require("fs");

const modelHeight = 640;
const modelWidth = 640;
const modelChannel = 3;
const CLS_CONF_IDX = 4;

async function preProcess(frame, dstInput) {
  const origData = new Uint8Array(frame.data);
  const hRatio = frame.height / modelHeight;
  const wRatio = frame.width / modelWidth;

  // resize data to model input size, uint8 data to float32 data,
  // and transpose from nhwc to nchw

  const origHStride = frame.width * 4;
  const origWStride = 4;
  var idx = 0;
  for (var c = 0; c < modelChannel; ++c) {
    for (var h = 0; h < modelHeight; ++h) {
      const origH = Math.round(h * hRatio);
      const origHOffset = origH * origHStride;

      for (var w = 0; w < modelWidth; ++w) {
        const origW = Math.round(w * wRatio);

        const origIndex = origHOffset + origW * origWStride + c;

        var val = origData[origIndex] / 255.0
        dstInput[idx] = val;

        idx++;
      }
    }
  }
}

function xywh2xyxy(cx, cy, width, height) {
  left = Math.round(cx - width/2.0);
  right = Math.round(cx + width/2.0);
  top = Math.round(cy - height/2.0);
  bottom = Math.round(cy + height/2.0);
  return [left, right, top, bottom];
}

function iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) {
  let [left1, right1, top1, bottom1] = xywh2xyxy(cx1, cy1, w1, h1);
  area1 = w1 * h1;

  let [left2, right2, top2, bottom2] = xywh2xyxy(cx2, cy2, w2, h2);
  area2 = w2 * h2;

  /* interaction */
  ll = Math.max(left1, left2);
  rr = Math.min(right1, right2);
  tt = Math.max(top1, top2);
  bb = Math.min(bottom1, bottom2);

  iw = Math.max(0, rr - ll);
  ih = Math.max(0, bb - tt);
  intersection_area = iw * ih;

  union_area = area1 + area2 - intersection_area;
  return intersection_area / union_area;
}

const YOLOV5S_BOXES = 25200;
const YOLOV5S_CLASSES = 85;
const CONF_THRES = 0.25;
const IOU_THRES = 0.45;

async function non_max_suppression(input) {
  var candidates = [];
  for (var idx = 0; idx < YOLOV5S_BOXES; idx++) {
    var arr = input.subarray(idx * YOLOV5S_CLASSES, (idx+1) * YOLOV5S_CLASSES);
    if (arr[CLS_CONF_IDX] > CONF_THRES) {
      candidates.push(arr);
    }
  }
  /* use Non Max Suppression to get 'selected' */
  candidates = candidates.sort((a, b) => b[CLS_CONF_IDX] - a[CLS_CONF_IDX]); /* sort by score */
  var selected = [];
  candidates.forEach(cand => {
    let add = true;
    for (var idx = 0; idx < selected.length; idx++) {
      const val = iou(cand[0], cand[1], cand[2], cand[3], selected[idx][0], selected[idx][1], selected[idx][2], selected[idx][3]);
      if (val > IOU_THRES) {
        add = false;
      }
    }

    if (add) {
      selected.push(cand);
    }
  });

  /* get simplified output from 'selected' */
  var answer = [];
  selected.forEach(sel => {
    max_cls_conf = 0;
    max_cls = null;
    for (var idx = 5; idx < YOLOV5S_CLASSES; idx++) {
      if (sel[idx] > max_cls_conf) {
        max_cls_conf = sel[idx];
        max_cls = idx-5;
      }
    }
    let [x1, x2, y1, y2] = xywh2xyxy(Math.round(sel[0]), Math.round(sel[1]), Math.round(sel[2]), Math.round(sel[3]))
    answer.push([x1, x2, y1, y2, max_cls]);
  });
  console.log(answer);
}

async function main() {
  inkjet.decode(fs.readFileSync("./bcd.jpg"), function(err, decoded) {
    if (decoded != undefined) {
      modelInput = new Float32Array(modelHeight*modelWidth*3);
      preProcess(decoded, modelInput);

      tensor = new ort.Tensor("float32", modelInput, [1, 3, modelHeight, modelWidth]);
      ort.InferenceSession.create("./yolov5m.onnx").then((se) => {
        se.run({"images": tensor}).then((res) => {
          non_max_suppression(res.output.data);
        });
      });
    }
  });

  console.log("Finish");
}

main();
