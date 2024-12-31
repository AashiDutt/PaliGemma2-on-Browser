import { AutoProcessor, PaliGemmaForConditionalGeneration } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.2.4';

// HTML elements
const imageUpload = document.getElementById('imageUpload');
const runButton = document.getElementById('runButton');
const imageContainer = document.getElementById('imageContainer');
const resultDiv = document.getElementById('result');

let model, processor;
let raw_image;

// Function to generate a random RGB color
function getRandomColor() {
  const r = Math.floor(Math.random() * 256);
  const g = Math.floor(Math.random() * 256);
  const b = Math.floor(Math.random() * 256);
  return `rgb(${r},${g},${b})`;
}

async function initializeModel() {
  const model_id = "onnx-community/paligemma2-3b-pt-224"; // Change this to use a different PaliGemma model
    resultDiv.innerHTML = "Loading model and processor...";
  try{
    processor = await AutoProcessor.from_pretrained(model_id);
    model = await PaliGemmaForConditionalGeneration.from_pretrained(
      model_id,
      {
        dtype: {
          embed_tokens: "q8", // or 'fp16'
          vision_encoder: "q8", // or 'q4', 'fp16'
          decoder_model_merged: "q4", // or 'q4f16'
        },
      }
    );
    resultDiv.innerHTML = "Model and processor loaded successfully!";
    runButton.disabled = false;

  } catch (error) {
     resultDiv.innerHTML = `Error loading model or processor: ${error.message}`;
     console.error("Error loading model or processor:", error);
  }
}

imageUpload.addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) {
        resultDiv.innerHTML = 'Please select an image.';
        return;
    }
  
    try {
        const imageURL = URL.createObjectURL(file);
        const img = new Image();
        img.src = imageURL;
        
        await new Promise((resolve) => {
            img.onload = resolve;
        });

       raw_image = img;
      
        imageContainer.innerHTML = '';
        imageContainer.appendChild(img);
  
        resultDiv.innerHTML = "Image loaded.";
        
    } catch (error) {
        resultDiv.innerHTML = `Error loading image: ${error.message}`;
        console.error("Error loading image:", error);
    }
});

runButton.addEventListener('click', async () => {
    if(!raw_image){
        resultDiv.innerHTML = 'No image loaded.';
        return;
    }

  const prompt = "<image>detect bounding box of car"; // Caption for detection

  // Extract the label from the prompt
    const labelMatch = prompt.match(/detect bounding box of (\w+)/);
    const label = labelMatch ? labelMatch[1] : "Unknown";
    const capitalizedLabel = label.charAt(0).toUpperCase() + label.slice(1);
  
    try {
        resultDiv.innerHTML = "Preparing inputs...";
      const inputs = await processor(raw_image, prompt);

        resultDiv.innerHTML = "Generating response from the model...";
        const response = await model.generate({
            ...inputs,
            max_new_tokens: 100,
        });

        const generatedIds = response.slice(null, [inputs.input_ids.dims[1], null]);

        const decodedAnswer = processor.batch_decode(generatedIds, {
            skip_special_tokens: true,
        });
        resultDiv.innerHTML = `Decoded answer: ${decodedAnswer[0]}`;

        const boundingBoxes = decodedAnswer[0].match(/<loc(\d+)>/g);
        
        if (boundingBoxes && boundingBoxes.length === 4) {
           const coordinates = boundingBoxes.map(tag => parseInt(tag.replace("<loc", "").replace(">", "")));
            const [y1, x1, y2, x2] = coordinates.map(coord => Math.floor(coord));

            const normX1 = Math.round((x1 / 1024) * raw_image.width);
            const normY1 = Math.round((y1 / 1024) * raw_image.height);
            const normX2 = Math.round((x2 / 1024) * raw_image.width);
            const normY2 = Math.round((y2 / 1024) * raw_image.height);

            resultDiv.innerHTML += `<br>Normalized Bounding Box: [${normX1}, ${normY1}, ${normX2}, ${normY2}]`;

           // Create Canvas for Drawing
            const canvas = document.createElement('canvas');
            canvas.width = raw_image.width;
            canvas.height = raw_image.height;
            const ctx = canvas.getContext('2d');

            ctx.drawImage(raw_image, 0, 0);

            const randomColor = getRandomColor();

            ctx.strokeStyle = randomColor;
            ctx.lineWidth = 5;
            ctx.strokeRect(normX1, normY1, normX2 - normX1, normY2 - normY1);
          
            const labelPadding = 10;
            const textWidth = ctx.measureText(capitalizedLabel).width;
            const labelWidth = textWidth * 2.5;
            const labelHeight = 30;
            const labelY = normY1 - labelHeight;

            ctx.fillStyle = randomColor;
            ctx.fillRect(normX1, labelY, labelWidth, labelHeight);

            ctx.fillStyle = "white";
            ctx.font = "bold 20px Arial";
            ctx.fillText(capitalizedLabel, normX1 + labelPadding, labelY + labelHeight - labelPadding);

             // Replace the image with the canvas drawing
             imageContainer.innerHTML = '';
             imageContainer.appendChild(canvas);

        }
         else {
            resultDiv.innerHTML += "<br>"+decodedAnswer[0];
        }

    } catch (error) {
          resultDiv.innerHTML = `Error generating response: ${error.message}`;
            console.error("Error generating response:", error);
    }
});

initializeModel();
