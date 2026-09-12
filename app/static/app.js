// Handles the form submit, the loading state and the result display.

const form = document.getElementById("predict_form");
const submitButton = document.getElementById("submit_button");
const propertyType = document.getElementById("property_type");
const areaLabel = document.getElementById("area_label");
const areaHint = document.getElementById("area_hint");
const warningEl = document.getElementById("result_warning");

const HOUSE_LIKE = ["House", "Townhouse", "Villa"];

const states = {
  empty: document.getElementById("state_empty"),
  loading: document.getElementById("state_loading"),
  error: document.getElementById("state_error"),
  result: document.getElementById("state_result"),
};

function showState(name) {
  Object.keys(states).forEach(function (key) {
    states[key].classList.toggle("is_hidden", key !== name);
  });
}

function formatMoney(value) {
  return "$" + Math.round(value).toLocaleString("en-AU");
}

// The sites report land area for a house and floor area for a flat, so
// the label follows whichever type is selected
function syncAreaLabel() {
  const isHouseLike = HOUSE_LIKE.includes(propertyType.value);
  areaLabel.textContent = isHouseLike ? "Land size" : "Floor area";
  areaHint.textContent = isHouseLike
    ? "Square metres of land. Optional, often not listed."
    : "Square metres of internal floor area. Optional, often not listed.";
}

propertyType.addEventListener("change", syncAreaLabel);
syncAreaLabel();

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  const payload = {
    suburb: document.getElementById("suburb").value,
    property_type: propertyType.value,
    sale_method: document.getElementById("sale_method").value,
    bedrooms: document.getElementById("bedrooms").value,
    bathrooms: document.getElementById("bathrooms").value,
    parking_spaces: document.getElementById("parking_spaces").value,
    area_sqm: document.getElementById("area_sqm").value,
  };

  submitButton.disabled = true;
  submitButton.textContent = "Estimating";
  showState("loading");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      document.getElementById("error_message").textContent =
        data.error || "The estimate could not be calculated.";
      showState("error");
      return;
    }

    document.getElementById("result_price").textContent = formatMoney(data.price);
    document.getElementById("result_range").textContent =
      "Likely range " + formatMoney(data.low) + " to " + formatMoney(data.high);
    document.getElementById("result_accuracy").textContent =
      "In " + data.suburb + " this model is typically within " +
      data.error_pct.toFixed(1) + " percent of the sale price, measured by " +
      "cross validation on sales it had not seen.";

    warningEl.textContent = data.warning || "";
    warningEl.classList.toggle("is_hidden", !data.warning);

    showState("result");
  } catch (error) {
    document.getElementById("error_message").textContent =
      "Could not reach the prediction service. Check that the app is still running.";
    showState("error");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Estimate price";
  }
});
