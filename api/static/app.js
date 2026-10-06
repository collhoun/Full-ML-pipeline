"use strict";

const SQFT_PER_SQM = 10.7639; // 1 м² = 10.7639 кв. фута
const MAPE = 0.107; // средняя абсолютная процентная ошибка модели

// Поля, значения которых задаются в квадратных метрах
const AREA_FIELDS = [
    "GrLivArea", "LotArea", "GarageArea", "LowQualFinSF",
    "OpenPorchSF", "EnclosedPorch", "WoodDeckSF", "ScreenPorch", "SsnPorch3", "PoolArea",
];

// Поля, которые нужно перевести в API-имена
const ALIASES = { SsnPorch3: "3SsnPorch" };

// Поля, у которых пустое значение равносильно отсутствию объекта (стратегия импутации «0»)
const ABSENCE_ZERO = new Set([
    "GarageArea", "BsmtFullBath", "BsmtHalfBath", "Fireplaces", "PoolArea",
    "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "SsnPorch3", "ScreenPorch",
]);

const REQUIRED = ["GrLivArea", "OverallQual", "YearBuilt", "LotArea", "OverallCond"];

const DEMO = {
    GrLivArea: 159, LotArea: 785, YearBuilt: 2003, OverallQual: 7, OverallCond: 5,
    FullBath: 2, BsmtFullBath: 1, KitchenAbvGr: 1, KitchenQual: "Gd",
    GarageType: "Attchd", GarageArea: 51, GarageQual: "TA", GarageCond: "TA",
    OpenPorchSF: 30, WoodDeckSF: 20, Fireplaces: 1,
    MSSubClass: "60", Foundation: "PConc", CentralAir: "Y",
    Electrical: "SBrkr", HeatingQC: "Gd", ExterQual: "Gd", ExterCond: "TA",
    BsmtQual: "Gd", BsmtCond: "TA",
};

const form = document.getElementById("house-form");
const els = {
    placeholder: document.getElementById("placeholder"),
    result: document.getElementById("result"),
    price: document.getElementById("price"),
    range: document.getElementById("range"),
    meterFill: document.getElementById("meter-fill"),
    meterNeedle: document.getElementById("meter-needle"),
    note: document.getElementById("note"),
    facts: document.getElementById("facts"),
    alert: document.getElementById("alert"),
    submit: document.getElementById("btn-submit"),
};

let unit = "sqm";

// ---------------------------------------------------------------- утилиты

const $ = (id) => document.getElementById(id);

const money = (v) => "$" + Math.round(v).toLocaleString("ru-RU");

function isFilled(el) {
    if (el.type === "range") return true;
    return el.value !== "" && el.value !== null;
}

function showAlert(msg) {
    els.alert.textContent = msg;
    els.alert.classList.remove("hidden");
}

function hideAlert() {
    els.alert.classList.add("hidden");
    els.alert.textContent = "";
}

function clearErrors() {
    document.querySelectorAll(".err").forEach((e) => (e.textContent = ""));
    document.querySelectorAll(".invalid").forEach((e) => e.classList.remove("invalid"));
}

function setError(name, msg) {
    const slot = document.querySelector(`[data-err="${name}"]`);
    if (slot) slot.textContent = msg;
    const input = $(name);
    if (input) input.classList.add("invalid");
}

// ---------------------------------------------------------------- единицы

function toSqft(sqm) {
    return Math.round(sqm * SQFT_PER_SQM * 100) / 100;
}

function syncUnits() {
    document.querySelectorAll(".unit").forEach((u) => (u.textContent = unit === "sqm" ? "м²" : "кв. фт"));
}

// ---------------------------------------------------------------- сбор данных

function collectPayload() {
    const payload = {};
    let absentCount = 0;

    for (const name of AREA_FIELDS) {
        const el = $(name);
        if (!el || !isFilled(el)) {
            absentCount++;
            continue;
        }
        const raw = parseFloat(el.value);
        if (Number.isNaN(raw)) {
            absentCount++;
            continue;
        }
        payload[ALIASES[name] || name] = unit === "sqm" ? toSqft(raw) : raw;
    }

    form.querySelectorAll("input, select").forEach((el) => {
        if (!el.name || el.tagName === "BUTTON" || el.type === "submit") return;
        if (AREA_FIELDS.includes(el.name)) return;
        if (el.type === "range") {
            payload[ALIASES[el.name] || el.name] = parseInt(el.value, 10);
            return;
        }
        if (!isFilled(el)) {
            absentCount++;
            return;
        }
        const name = ALIASES[el.name] || el.name;
        const num = parseFloat(el.value);
        payload[name] = el.tagName === "SELECT" ? el.value : (Number.isNaN(num) ? el.value : num);
    });

    return { payload, absentCount };
}

// ---------------------------------------------------------------- валидация

function validate() {
    clearErrors();
    const problems = [];

    const checks = [
        ["GrLivArea", "Укажите жилую площадь", v => v > 0],
        ["LotArea", "Укажите площадь участка", v => v > 0],
        ["YearBuilt", "Укажите год постройки", v => v > 1800 && v <= 2030],
    ];

    for (const [name, msg, ok] of checks) {
        const el = $(name);
        const raw = el.value === "" ? NaN : parseFloat(el.value);
        if (!ok(raw)) {
            setError(name, msg);
            problems.push(msg);
        }
    }

    return problems;
}

// ---------------------------------------------------------------- запрос

async function requestPrediction() {
    const problems = validate();
    if (problems.length) {
        showAlert("Проверьте обязательные поля: " + problems.join("; ") + ".");
        return;
    }
    hideAlert();
    els.submit.disabled = true;
    els.submit.textContent = "Считаем…";

    try {
        const { payload, absentCount } = collectPayload();
        const resp = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (resp.status === 422) {
            const err = await resp.json().catch(() => null);
            showAlert("Сервер отклонил данные: " + describe422(err));
            return;
        }
        if (!resp.ok) {
            const err = await resp.json().catch(() => null);
            showAlert((err && err.detail) || `Ошибка сервера (${resp.status}).`);
            return;
        }

        const data = await resp.json();
        renderResult(data.predicted_price, payload, absentCount);
    } catch (e) {
        showAlert("Не удалось связаться с сервером. Проверьте, что API запущен.");
    } finally {
        els.submit.disabled = false;
        els.submit.textContent = "Рассчитать стоимость";
    }
}

function describe422(err) {
    if (!err || !Array.isArray(err.detail)) return "неверные значения полей";
    return err.detail
        .map(d => {
            const loc = Array.isArray(d.loc) ? d.loc.filter(x => x !== "body").join(".") : "";
            return loc ? `${loc}: ${d.msg}` : d.msg;
        })
        .join("; ");
}

// ---------------------------------------------------------------- отрисовка

function renderResult(price, payload, absentCount) {
    els.placeholder.classList.add("hidden");
    els.result.classList.remove("hidden");

    els.price.textContent = money(price);

    const low = price * (1 - MAPE);
    const high = price * (1 + MAPE);
    els.range.textContent = `Вероятный диапазон: ${money(low)} — ${money(high)}`;

    const left = (low / (high * 1.15)) * 100;
    const width = Math.max(((high - low) / (high * 1.15)) * 100, 4);
    els.meterFill.style.left = left + "%";
    els.meterFill.style.width = width + "%";
    els.meterNeedle.style.left = ((price / (high * 1.15)) * 100) + "%";

    const perSqm = price / payload.GrLivArea;
    els.note.textContent =
        `≈ ${money(perSqm)} за кв. фут жилой площади. ` +
        (absentCount > 0
            ? `Не указано параметров: ${absentCount} — они заменены статистическими значениями по обучающей выборке, точность оценки может снизиться.`
            : "Заполнены все параметры — оценка выполнена по полному набору признаков.");

    renderFacts(payload, price);
}

function renderFacts(payload, price) {
    const rows = [];
    const per = (v) => money(v);

    rows.push(["Жилая площадь", per(payload.GrLivArea) + " кв. фт", true]);
    rows.push(["Площадь участка", per(payload.LotArea) + " кв. фт", true]);
    rows.push(["Год постройки", payload.YearBuilt, true]);
    rows.push(["Качество / состояние", `${payload.OverallQual} / ${payload.OverallCond}`, true]);
    rows.push(["Цена за кв. фут", per(price / payload.GrLivArea), false]);

    if (payload.GarageType !== undefined || payload.GarageArea !== undefined) {
        rows.push(["Гараж", garageLabel(payload), payload.GarageType !== undefined]);
    }
    if (payload.CentralAir !== undefined) {
        rows.push(["Кондиционер", payload.CentralAir === "Y" ? "есть" : "нет", true]);
    }
    if (payload.Foundation !== undefined) {
        const opt = $( "Foundation" ).selectedOptions[0];
        rows.push(["Фундамент", opt ? opt.textContent : payload.Foundation, true]);
    }

    els.facts.innerHTML = "";
    for (const [name, value, filled] of rows) {
        const div = document.createElement("div");
        div.className = "fact" + (filled ? " is-filled" : "");
        const s = document.createElement("span");
        s.textContent = name;
        const b = document.createElement("b");
        b.textContent = value;
        div.append(s, b);
        els.facts.appendChild(div);
    }
}

function garageLabel(payload) {
    const sel = $("GarageType");
    const typeText = sel && sel.value ? sel.selectedOptions[0].textContent : "не указан";
    const area = payload.GarageArea ? `, ${payload.GarageArea} кв. фт` : "";
    return typeText + area;
}

// ---------------------------------------------------------------- счётчики аккордеонов

function countFilled(acc) {
    let n = 0;
    acc.querySelectorAll("input, select").forEach((el) => {
        if (el.name && el.type !== "submit" && isFilled(el)) n++;
    });
    const slot = acc.querySelector(".acc-count");
    if (slot) slot.textContent = n ? `заполнено: ${n}` : "";
}

// ---------------------------------------------------------------- demo / reset

function fillDemo() {
    for (const [name, value] of Object.entries(DEMO)) {
        const el = $(name);
        if (el) el.value = String(value);
    }
    onAnyChange();
}

function resetForm() {
    form.reset();
    document.querySelectorAll(".err").forEach((e) => (e.textContent = ""));
    document.querySelectorAll(".invalid").forEach((e) => e.classList.remove("invalid"));
    hideAlert();
    syncRanges();
    onAnyChange();
}

// ---------------------------------------------------------------- события

function syncRanges() {
    document.querySelectorAll('input[type="range"]').forEach((r) => {
        const out = document.querySelector(`[data-out="${r.name}"]`);
        if (out) out.textContent = r.value;
    });
}

function onAnyChange() {
    syncRanges();
    document.querySelectorAll("details.acc").forEach(countFilled);
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    requestPrediction();
});

form.addEventListener("input", () => {
    hideAlert();
    onAnyChange();
});

form.addEventListener("change", onAnyChange);

$("btn-demo").addEventListener("click", fillDemo);
$("btn-reset").addEventListener("click", resetForm);

document.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".seg").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        unit = btn.dataset.unit;
        syncUnits();
    });
});

onAnyChange();