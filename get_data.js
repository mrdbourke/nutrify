import { fdc_ids_string_index } from "./constants.js";
import { supabase } from "./supabaseClient.js";

// Setup variables
var isThisCorrect, yesButton, noButton, target_food_code;

// Get all data in one hit from Supabase (this is all rows in the DB)
export async function get_all_food_data_from_supabase() {
    let { data, error } = await supabase
        .from("food_data_central_nutrition_data_test")
        .select("*")

    if (error) {
        console.log(error);
        return;
    }
    // Log data if necessary
    // console.log(data);
    return data;
};

// Make a function to get food data and then update the DOM elements
export async function getFoodData(food_selection, data) {
    // Start timer
    // var startTime = performance.now()

    console.log("Logging data again...")
    console.log(data);
    console.log(fdc_ids_string_index);
    target_food_code = fdc_ids_string_index[food_selection.toLowerCase()];
    console.log(`Target food code: ${target_food_code}`)
    const target_food_data = data.find(element => element["fdc_id"] == target_food_code)
    // const data = await get_food_data_from_supabase(target_food_code);
    // You can get data from "data" using JavaScript destructuring
    console.log(target_food_data);
    const food_name = target_food_data["food"];
    console.log(food_name);
    // document.getElementById("food_name").textContent = food_name;

    // Get nutrient values and update macronutrients
    const protein_amount = target_food_data["protein"];
    console.log(`Protein amount: ${protein_amount}`)
    document.getElementById("protein_amount").textContent = protein_amount + "g";

    const carbohydrate_amount = target_food_data["carbohydrate_by_difference"];
    console.log(`Carbohydrate amount: ${carbohydrate_amount}`)
    document.getElementById("carbohydrate_amount").textContent = carbohydrate_amount + "g";

    const fat_amount = target_food_data["total_lipid_fat"];
    console.log(`Fat amount: ${fat_amount}`)
    document.getElementById("fat_amount").textContent = fat_amount + "g";

    // End timer
    // var endTime = performance.now()

    // Log time
    // console.log(`Call to getFoodData took ${endTime - startTime} milliseconds`)
    // document.getElementById("time_taken").textContent = `${(endTime - startTime) / 1000} seconds`
}

// Make a function to log to Supabase which food was display and whether the information was correct or not
// Could add: uuid, timestamp, food_id, correct: yes/no
// Want to also make sure the button can only be pressed once (e.g. the functionality gets removed once its been clicked)
// Or is this bad? What if someone wants to say, "no it's not correct?"... that can come later
// Workflow: click the button, something gets logged to Supabase, "thank you" message appears and buttons disappear?
export async function updateIsThisCorrect(uuid, is_correct, fdc_id) {
    let { data, error } = await supabase
        .from("nutrify_is_this_correct")
        .insert([
            {
                id: uuid,
                is_correct: is_correct,
                pred_fdc_id: fdc_id
            }
        ],
            {
                returning: 'minimal'
            })
    console.log(`Updating Supabase with correct: ${is_correct} for ${fdc_id} with UUID: ${uuid}`)
};

// Function update information on Supabase when a "is_this_correct?" button pressed
export function isThisCorrectButtonClicked(is_correct, uuid) {
    if (is_correct) {
        console.log("Clicking the 'yes' button")
    } else {
        console.log("Clicking the 'no' button")
    }

    // Update relative variables
    var fdc_id = target_food_code;

    // Update table
    updateIsThisCorrect(
        uuid,
        is_correct,
        fdc_id
    )

    // Hide buttons/"is this correct?" text after one is clicked
    yesButton.style.display = "none";
    noButton.style.display = "none";
    isThisCorrect.style.display = "none";

    // Show "Thank you for your feedback" message.
    showThankYouMessage(is_correct);
}

// Function to show "thank you" message based on what option is selected
export function showThankYouMessage(is_correct) {
    // Get "Thank you message" text
    var thankYouMessage = document.getElementById("thank_you_message");

    // Update text based on whether correct or not
    if (is_correct) {
        thankYouMessage.textContent = "Nice! Thank you for letting us know."
        thankYouMessage.style.display = "block";
    } else {
        thankYouMessage.textContent = "Dam. Looks like there's room for improvement, thank you for letting us know."
        thankYouMessage.style.display = "block";
    }
}

// Make a function to update the DOM with yes/no buttons as the food data gets fetched
export async function showCorrectButtons(uuid) {
    // Get "is_this_correct" text and show it
    isThisCorrect = document.getElementById("is_this_correct");
    isThisCorrect.style.display = "block";

    // Get "Yes" button and show it
    yesButton = document.getElementById("yes_button");
    yesButton.style.display = "inline";

    // If the "Yes" button is clicked...
    yesButton.onclick = function () {
        isThisCorrectButtonClicked(true, uuid)
    };

    // Get "No" button and display it
    noButton = document.getElementById("no_button");
    noButton.style.display = "inline";

    // If the "No" button is clicked...
    noButton.onclick = function () {
        isThisCorrectButtonClicked(false, uuid)
    };
}
