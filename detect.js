const ort = require("onnxruntime-node");
const inkjet = require("inkjet");
const fs = require("fs");

const modelHeight = 640;
const modelWidth = 640;
const modelChannel = 3;

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

        var offset;
        if (c == 0) {
          offset = 2;
        } else if (c == 2) {
          offset = 0;
        } else {
          offset = c;
        }
        const origIndex = origHOffset + origW * origWStride + offset;

        var val = origData[origIndex] / 255.0
        dstInput[idx] = val;

        idx++;
      }
    }
  }
}

function xywh2xyxy(cx1, cy1, w1, h1) {
  left1 = Math.round(cx1 - w1/2.0);
  right1 = Math.round(cx1 + w1/2.0);
  top1 = Math.round(cy1 - h1/2.0);
  bottom1 = Math.round(cy1 + h1/2.0);
  console.log("show:", left1, right1, top1, bottom1);
}

function iou(cx1, cy1, w1, h1, cx2, cy2, w2, h2) {
  left1 = Math.round(cx1 - w1/2.0);
  right1 = Math.round(cx1 + w1/2.0);
  top1 = Math.round(cy1 - h1/2.0);
  bottom1 = Math.round(cy1 + h1/2.0);
  area1 = w1 * h1;

  left2 = Math.round(cx2 - w2/2.0);
  right2 = Math.round(cx2 + w2/2.0);
  top2 = Math.round(cy2 - h2/2.0);
  bottom2 = Math.round(cy2 + h2/2.0);
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
  ret = intersection_area / union_area;
  console.log(left1, right1, top1, bottom1);
  console.log(left2, right2, top2, bottom2, ret);
  console.log(intersection_area, union_area);
  return ret;
}

const YOLOV5S_BOXES = 25200;
const YOLOV5S_CLASSES = 85;
const CONF_THRES = 0.25;
const IOU_THRES = 0.45;

async function non_max_suppression(input) {
  var candidates = [];
  for (var idx = 0; idx < YOLOV5S_BOXES; idx++) {
    var arr = input.subarray(idx * YOLOV5S_CLASSES, (idx+1) * YOLOV5S_CLASSES);
    max_cls_conf = 0;
    max_cls = null;
    for (var jdx = 5; jdx < YOLOV5S_CLASSES; jdx++) {
      if (arr[jdx] > max_cls_conf) {
        max_cls_conf = arr[jdx];
        max_cls = jdx-5;
      }
    }
    //arr[4] = arr[4] * max_cls_conf;
    if (arr[4] > CONF_THRES) {
      candidates.push(arr);
      console.log(max_cls);
    }
  }
  candidates = candidates.sort((a, b) => b[4] - a[4]); /* sort by score */
  console.log("candidates:", candidates);
  var selected = [];
  candidates.forEach(cand => {
    xywh2xyxy(cand[0], cand[1], cand[2], cand[3]);
    let add = true;
    for (var idx = 0; idx < selected.length; idx++) {
      const val = iou(cand[0], cand[1], cand[2], cand[3], selected[idx][0], selected[idx][1], selected[idx][2], selected[idx][3]);
      console.log(val);
      if (val > IOU_THRES) {
        add = false;
        console.log("add false");
      }
    }

    if (add) {
      console.log("add");
      selected.push(cand);
    }
  })
  console.log(selected);
}

async function main() {

  console.log("Start");

  inkjet.decode(fs.readFileSync("./bluejay640.jpg"), function(err, decoded) {
    if (decoded != undefined) {
      modelInput = new Float32Array(modelHeight*modelWidth*3);
      preProcess(decoded, modelInput);

      tensor = new ort.Tensor("float32", modelInput, [1, 3, modelHeight, modelWidth]);
      console.log("go", tensor);
      ort.InferenceSession.create("./yolov5m.onnx").then((se) => {
        console.log("session", se.inputNames);
        se.run({"images": tensor}).then((res) => {
          non_max_suppression(res.output.data);
        });
      });
    }
  });

  console.log("Finish");
}

main();
