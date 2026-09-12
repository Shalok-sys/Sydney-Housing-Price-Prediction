// Handles the form submit, the loading state and the result display.

const form = document.getElementById("predict_form");
const submitButton = document.getElementById("submit_button");
const propertyType = document.getElementById("property_type");
const landInput = document.getElementById("land_size_sqm");

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

// Units have no land of their own, so switch that input off
function syncLandInput() {
  const isUnit = propertyType.value === "Unit";
  landInput.disabled = isUnit;
  if (isUnit) {
    landInput.value = "";
  } else if (landInput.value === "") {
    landInput.value = "450";
  }
}

propertyType.addEventListener("change", syncLandInput);
syncLandInput();

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  const payload = {
    suburb: document.getElementById("suburb").value,
    property_type: propertyType.value,
    bedrooms: document.getElementById("bedrooms").value,
    bathrooms: document.getElementById("bathrooms").value,
    parking_spaces: document.getElementById("parking_spaces").value,
    land_size_sqm: landInput.value,
    floor_area_sqm: document.getElementById("floor_area_sqm").value,
    distance_to_cbd_km: document.getElementById("distance_to_cbd_km").value,
    distance_to_station_km: document.getElementById("distance_to_station_km").value,
    distance_to_school_km: document.getElementById("distance_to_school_km").value,
    year_built: document.getElementById("year_built").value,
    days_on_market: document.getElementById("days_on_market").value,
    renovated: document.getElementById("renovated").checked,
    has_pool: document.getElementById("has_pool").checked,
    agent_description: document.getElementById("agent_description").value,
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
      "cross validation on the training data.";
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
