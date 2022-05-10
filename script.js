// Imports
import { fdc_ids_as_array } from "./constants.js";
import { getFoodData, get_all_food_data_from_supabase, showCorrectButtons } from "./get_data.js";
import { uuidv4 } from "./utils.js"

// Loading data
console.log(`Loaded data for following FDC IDs: ${fdc_ids_as_array}`);

// Check to see if TF.js is available
console.log(`Loaded TensorFlow.js - version: ${tf.version.tfjs}`);

// Image uploading
const fileInput = document.getElementById("file-input");
const image = document.getElementById("image");
const uploadButton = document.getElementById("upload-button");

// Var creation
var uuid;

// Get all food data in one hit from Supabase and save it to a constant
const data = await get_all_food_data_from_supabase();
console.log("Logging data:")
console.log(data);

// Function to get image
function getImage() {
    // Throw error if file not found
    if (!fileInput.files[0]) throw new Error("Image not found");
    const file = fileInput.files[0];

    // Hide thank you message (if it's on show)
    var thankYouMessage = document.getElementById("thank_you_message")
    thankYouMessage.style.display = "none";

    // Get the data url from the image
    const reader = new FileReader();

    // When reader is ready display image
    reader.onload = function (event) {
        // Get data URL
        const dataUrl = event.target.result;

        // Create image object
        const imageElement = new Image();
        imageElement.src = dataUrl;

        // Create UUID for image instance
        uuid = uuidv4();
        console.log(`UUID: ${uuid}`);

        // When image object loaded
        imageElement.onload = function () {
            // Display image
            image.setAttribute("src", this.src);

            // Log image parameters
            const currImage = tf.browser.fromPixels(imageElement);

            // Start timer
            var startTime = performance.now()

            // Classify image uploaded - 1st: to food/not food, 2nd: what food is it?
            // If the following outputs True, run with the food prediction,
            // if not, post a message saying no food found, please try another.
            if (foodNotFood(foodNotFoodModel, currImage)) {
                classifyImage(foodVisionModel, currImage);
            } else {
                // Update HTML to reflect no food
                predicted_class.textContent = "No food found, please try another image."
                protein_amount.textContent = ""
                carbohydrate_amount.textContent = ""
                fat_amount.textContent = ""
            }

            // Finish timer and output time of classification
            var endTime = performance.now()
            document.getElementById("time_taken").textContent = `${((endTime - startTime) / 1000).toFixed(4)} seconds`
        };

        document.body.classList.add("image-loaded");
    };

    // Get data url
    reader.readAsDataURL(file);
}

// Add listener to see if someone uploads an image
fileInput.addEventListener("change", getImage);
uploadButton.addEventListener("click", () => fileInput.click());

// Setup the model(s) code
let foodVisionModel; // This is in global scope
let foodNotFoodModel;

const foodVisionModelStringPath = "models/2022-01-16-nutrify_model_100_foods_manually_cleaned_10_classes_foods_v1.tflite"
const foodNotFoodModelStringPath = "models/2022-03-18_food_not_food_model_efficientnet_lite0_v1.tflite"

const loadModel = async () => {
    // Load foodVisionModel (predicts what food is in an image)
    // and foodNotFoodModel (predicts whether their is food in an image or not)
    try {
        const foodVisionTFLiteModel = await tflite.loadTFLiteModel(
            foodVisionModelStringPath
        );
        const foodNotFoodTFLiteModel = await tflite.loadTFLiteModel(
            foodNotFoodModelStringPath
        );

        // Set models to global scope
        foodVisionModel = foodVisionTFLiteModel; // assigning it to the global scope model as tfliteModel can only be used within this scope
        console.log(`Loaded model: ${foodVisionModelStringPath}`)

        foodNotFoodModel = foodNotFoodTFLiteModel
        console.log(`Loaded model: ${foodNotFoodModelStringPath}`)

    } catch (error) {
        console.log(error);
    }
};

// Load model and data
loadModel();

// Function to classify image
function classifyImage(model, image) {
    // Preprocess image
    image = tf.image.resizeBilinear(image, [240, 240]); // image size needs to be same as model inputs - EffNetB1 takes 240x240
    image = tf.expandDims(image);

    // Log image and model if needed
    // console.log(image);
    // console.log(model);

    // console.log(tflite.getDTypeFromTFLiteType("uint8")); // Gives int32 as output thus we cast int32 in below line
    console.log("Converting image to different datatype...");
    image = tf.cast(image, "int32"); // Model requires uint8
    console.log("Model about to predict what kind of food it is...");
    const output = model.predict(image);
    const output_values = tf.softmax(output.arraySync()[0]);

    console.log("Output of model:");
    console.log(output.arraySync()[0]); // arraySync() Returns an array to use

    console.log("After calling softmax on the output:");
    console.log(output_values.arraySync());

    // Update HTML
    const predicted_class_string = fdc_ids_as_array[output_values.argMax().arraySync()];
    predicted_class.textContent = predicted_class_string;
    // predicted_prob.textContent = output_values.max().arraySync() * 100 + "%";

    // Get data from Supabase and update HTML
    getFoodData(predicted_class_string, data);

    // Show "is this correct?" buttons
    showCorrectButtons(uuid);
}


// Function to classify whether the image is of food or not
function foodNotFood(model, image) {

    // Preprocess image
    image = tf.image.resizeBilinear(image, [224, 224]); // image size needs to be same as model inputs - EffNetB0 takes 224x224
    image = tf.expandDims(image);

    // console.log(tflite.getDTypeFromTFLiteType("uint8")); // Gives int32 as output thus we cast int32 in below line
    console.log("Converting image to different datatype...");
    image = tf.cast(image, "int32"); // Model requires uint8
    console.log("Model predicting food/not food...");

    // Make prediction on image
    const output = model.predict(image);

    // Calculate various values
    const output_values = tf.softmax(output.arraySync()[0]);
    const output_max = tf.max(output.arraySync()[0]);

    console.log("Output of foodNotFood model:");
    console.log(output.arraySync()[0]); // arraySync() Returns an array to use

    console.log("After calling softmax on the output:");
    console.log(output_values.arraySync());

    // Find out "food" or "not food" status
    const foodNotFoodClasses = {
        0: "Food",
        1: "Not Food"
    }

    const foodOrNot = output_values.argMax().arraySync()
    const foodOrNotPredProb = (((1 / 256) * output_max.arraySync()) * 100).toFixed(2)
    console.log(`Uploaded image predicted to be: ${foodNotFoodClasses[foodOrNot]}`)
    console.log(`Prediction probability of ${foodNotFoodClasses[foodOrNot]}: ${foodOrNotPredProb}%`);

    // Return 0 for "food" or 1 for "not food"
    if (foodOrNot == 0) {
        return true
    } else {
        return false
    }
}
