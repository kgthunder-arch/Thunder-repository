const fileInput = document.querySelector('input[type="file"][name="dataset"]');
const fileName = document.getElementById("file-name");
const testSize = document.getElementById("test_size");
const testSizeValue = document.getElementById("test-size-value");

if (fileInput && fileName) {
    fileInput.addEventListener("change", (event) => {
        const selected = event.target.files && event.target.files[0];
        fileName.textContent = selected ? `${selected.name} selected` : "No custom file selected.";
    });
}

if (testSize && testSizeValue) {
    const updateTestLabel = () => {
        testSizeValue.textContent = `${Math.round(Number(testSize.value) * 100)}%`;
    };

    updateTestLabel();
    testSize.addEventListener("input", updateTestLabel);
}

const resizePlotlyCharts = () => {
    if (!window.Plotly) {
        return;
    }

    document.querySelectorAll(".js-plotly-plot").forEach((chart) => {
        window.Plotly.Plots.resize(chart);
    });
};

window.addEventListener("load", resizePlotlyCharts);
window.addEventListener("resize", resizePlotlyCharts);
window.addEventListener("orientationchange", resizePlotlyCharts);
