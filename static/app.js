console.log("app.js loaded");

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM loaded");

    const uploadForm = document.getElementById("uploadForm");
    const imageInput = document.getElementById("imageInput");
    const dropZone = document.getElementById("dropZone");
    const previewWrapper = document.getElementById("previewWrapper");
    const previewImage = document.getElementById("previewImage");
    const submitButton = document.getElementById("submitButton");
    const debugText = document.getElementById("debugText");

    const resultSection = document.getElementById("resultSection");
    const resultTitle = document.getElementById("resultTitle");
    const confidenceValue = document.getElementById("confidenceValue");
    const statusBadge = document.getElementById("statusBadge");
    const recommendationText = document.getElementById("recommendationText");
    const topPredictions = document.getElementById("topPredictions");

    let selectedFile = null;

    function setDebug(message) {
        if (debugText) {
            debugText.textContent = message;
        }
        console.log(message);
    }

    function setFile(file) {
        if (!file) {
            setDebug("Файл не обрано.");
            return;
        }

        if (!file.type.startsWith("image/")) {
            setDebug("Обраний файл не є зображенням.");
            alert("Оберіть файл зображення у форматі JPG або PNG.");
            return;
        }

        selectedFile = file;

        setDebug(`Обрано файл: ${file.name}`);
        dropZone.classList.add("compact");

        const reader = new FileReader();

        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewWrapper.classList.remove("hidden");
        };

        reader.readAsDataURL(file);
    }

    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];
        setFile(file);
    });

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("dragover");

        const file = event.dataTransfer.files[0];
        setFile(file);
    });

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!selectedFile) {
            setDebug("Спочатку оберіть зображення.");
            alert("Спочатку оберіть зображення.");
            return;
        }

        const formData = new FormData();
        formData.append("file", selectedFile);

        submitButton.disabled = true;
        submitButton.textContent = "Класифікація...";
        setDebug("Надсилання зображення на сервер...");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            console.log("Server response:", data);

            if (!response.ok) {
                showError(data.message || "Помилка під час обробки зображення.");
                return;
            }

            showResult(data);
            setDebug("Класифікацію завершено.");
        } catch (error) {
            console.error(error);
            showError("Не вдалося виконати запит до сервера.");
            setDebug("Помилка запиту до сервера.");
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = "Класифікувати зображення";
        }
    });

    function showError(message) {
        resultSection.classList.remove("hidden");

        resultTitle.textContent = "Помилка";
        confidenceValue.textContent = "—";
        recommendationText.textContent = message;

        statusBadge.textContent = "Помилка";
        statusBadge.className = "status-badge danger";

        topPredictions.innerHTML = "";
    }

    function showResult(data) {
        resultSection.classList.remove("hidden");

        resultTitle.textContent = data.display_name || data.predicted_class || "Невизначено";

        const confidence = data.confidence_percent ?? 0;
        confidenceValue.textContent = `${Number(confidence).toFixed(1)}%`;

        recommendationText.textContent = data.recommendation || data.message || "Рекомендація відсутня.";

        if (data.status === "success") {
            statusBadge.textContent = "Успішна класифікація";
            statusBadge.className = "status-badge";
        } else if (data.status === "uncertain") {
            statusBadge.textContent = "Недостатня впевненість";
            statusBadge.className = "status-badge warning";
        } else if (data.status === "bad_quality") {
            statusBadge.textContent = "Низька якість зображення";
            statusBadge.className = "status-badge danger";
        } else {
            statusBadge.textContent = "Результат";
            statusBadge.className = "status-badge";
        }

        topPredictions.innerHTML = "";

        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach((item) => {
                const percent = item.confidence_percent ?? item.confidence * 100;

                const element = document.createElement("div");
                element.className = "top-item";

                element.innerHTML = `
                    <div class="top-item-header">
                        <span>${item.display_name || item.class}</span>
                        <span>${Number(percent).toFixed(1)}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-fill" style="width: ${percent}%"></div>
                    </div>
                `;

                topPredictions.appendChild(element);
            });
        }
    }

    setDebug("Інтерфейс готовий до завантаження фото.");
});