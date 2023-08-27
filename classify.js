const ort = require("onnxruntime-node");
const inkjet = require("inkjet");
const fs = require("fs");

const modelHeight = 300;
const modelWidth = 300;
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

async function main() {

  console.log("Start");

  inkjet.decode(fs.readFileSync("./dabailu.jpg"), function(err, decoded) {
    if (decoded != undefined) {
      modelInput = new Float32Array(300*300*3);
      preProcess(decoded, modelInput);

      tensor = new ort.Tensor("float32", modelInput, [1, 3, 300, 300]);
      console.log("go", tensor);
      ort.InferenceSession.create("./bird_img_convnextv2_nano_20230720.onnx").then((se) => {
        console.log("session", se.inputNames);
        se.run({"input": tensor}).then((res) => {
          let num = new Float32Array(res.output.data)
          var max = num[0];
          var index = 0;
          for (var idx = 1; idx < num.length; idx++) {
            if (max < num[idx]) {
              max = num[idx];
              index = idx;
            }
          }
          console.log(index);
        });
      });
    }
  });

  console.log("Finish");
}

main();
