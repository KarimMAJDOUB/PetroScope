document.addEventListener('DOMContentLoaded', function () {
    const reservoirSelect = document.getElementById('reservoir-select');
    const rateSelect = document.getElementById('rate-select');
    const modelSelect = document.getElementById('model-select');
    const reservoirImage = document.getElementById('reservoir-image');
    const ctx = document.getElementById('oilRateChart').getContext('2d');
    const tableBody = document.querySelector('#data-table tbody');

    // Navigation Elements
    const navLinks = document.querySelectorAll('.sidebar-nav a');
    const pages = document.querySelectorAll('.page-content');

    // Filter Elements
    const filterDateMin = document.getElementById('filter-date-min');
    const filterDateMax = document.getElementById('filter-date-max');
    const filterActualMin = document.getElementById('filter-actual-min');
    const filterActualMax = document.getElementById('filter-actual-max');
    const filterPredictedMin = document.getElementById('filter-predicted-min');
    const filterPredictedMax = document.getElementById('filter-predicted-max');
    const filterStatus = document.getElementById('filter-status');
    const applyFiltersBtn = document.getElementById('apply-filters');
    const resetFiltersBtn = document.getElementById('reset-filters');

    // Chart Configuration
    Chart.defaults.color = '#e2e8f0';
    Chart.defaults.borderColor = '#334155';

    // Initial Data
    const initialData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        datasets: [
            {
                label: 'Actual',
                data: [1200, 1150, 1250, 1300, 1280, 1320],
                borderColor: '#3b82f6', // Blue
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4
            },
            {
                label: 'Predicted',
                data: [1220, 1180, 1240, 1290, 1310, 1350],
                borderColor: '#10b981', // Green
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderDash: [5, 5],
                tension: 0.4
            }
        ]
    };

    let mainChart = new Chart(ctx, {
        type: 'line',
        data: initialData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Oil Rate (bbl/d)',
                    color: '#f8fafc',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        color: '#e2e8f0'
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: '#334155'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: {
                        color: '#334155'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });

    // Model coefficients for different prediction accuracies
    const modelCoefficients = {
        'model1': { variance: 0.03, bias: 1.02 },  // XGBoost - very accurate
        'model2': { variance: 0.05, bias: 1.01 },  // Random Forest - accurate
        'model3': { variance: 0.08, bias: 0.98 },  // LSTM - moderate
        'model4': { variance: 0.06, bias: 1.03 },  // Transformer - good
        'model5': { variance: 0.12, bias: 0.95 }   // Linear Regression - less accurate
    };

    let currentModel = 'model1'; // Track current model
    let baseActualData = {}; // Store base actual data per reservoir
    let chartActualData = {}; // Store chart actual data per reservoir/rate combination

    // Generate Base Actual Data (doesn't change with model)
    function generateBaseActualData(reservoir, count = 100) {
        const data = [];
        const baseRate = reservoir === 'reservoir1' ? 1200 : (reservoir === 'reservoir2' ? 900 : 1500);
        const startDate = new Date('2023-01-01');

        for (let i = 0; i < count; i++) {
            const currentDate = new Date(startDate);
            currentDate.setDate(startDate.getDate() + i);
            const actual = Math.floor(baseRate + (Math.random() - 0.5) * 100);

            data.push({
                date: currentDate.toISOString().split('T')[0],
                reservoir: reservoir.charAt(0).toUpperCase() + reservoir.slice(1),
                actual: actual
            });
        }
        return data;
    }

    // Generate Table Data with predictions based on model
    function generateTableData(reservoir, model = 'model1') {
        // Get or create base actual data for this reservoir
        if (!baseActualData[reservoir]) {
            baseActualData[reservoir] = generateBaseActualData(reservoir, 150);
        }

        const modelCoef = modelCoefficients[model];
        const data = baseActualData[reservoir].map(row => {
            const predicted = Math.floor(row.actual * modelCoef.bias + (Math.random() - 0.5) * (row.actual * modelCoef.variance));
            const status = Math.abs(row.actual - predicted) > 80 ? 'Check' : 'Normal';

            return {
                date: row.date,
                reservoir: row.reservoir,
                actual: row.actual,
                predicted: predicted,
                status: status
            };
        });

        return data;
    }

    let allTableData = []; // Store all generated data

    function populateTable(data) {
        tableBody.innerHTML = '';
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.date}</td>
                <td>${row.reservoir}</td>
                <td>${row.actual} bbl/d</td>
                <td>${row.predicted} bbl/d</td>
                <td><span class="${row.status === 'Normal' ? 'status-ok' : 'status-warning'}">${row.status}</span></td>
            `;
            tableBody.appendChild(tr);
        });
    }

    // Update Chart Data
    function updateChart(reservoir, rateType, model = 'model1') {
        const chartKey = `${reservoir}-${rateType}`;

        // Generate or retrieve actual data (stays the same for this reservoir/rate combination)
        if (!chartActualData[chartKey]) {
            let baseValue = 1000;
            if (reservoir === 'reservoir2') baseValue = 800;
            if (reservoir === 'reservoir3') baseValue = 1400;

            if (rateType === 'Gas') baseValue = baseValue * 5;
            if (rateType === 'Water') baseValue = baseValue * 0.5;

            chartActualData[chartKey] = Array.from({ length: 6 }, () =>
                Math.floor(baseValue + (Math.random() - 0.5) * (baseValue * 0.1))
            );
        }

        // Use stored actual data
        const newDataActual = chartActualData[chartKey];

        // Generate predicted data based on model (changes with model)
        const modelCoef = modelCoefficients[model];
        let baseValue = 1000;
        if (reservoir === 'reservoir2') baseValue = 800;
        if (reservoir === 'reservoir3') baseValue = 1400;
        if (rateType === 'Gas') baseValue = baseValue * 5;
        if (rateType === 'Water') baseValue = baseValue * 0.5;

        const newDataPredicted = newDataActual.map(actual =>
            Math.floor(actual * modelCoef.bias + (Math.random() - 0.5) * (baseValue * modelCoef.variance))
        );

        mainChart.data.datasets[0].data = newDataActual;
        mainChart.data.datasets[1].data = newDataPredicted;

        let unit = 'bbl/d';
        if (rateType === 'Gas') unit = 'mcf/d';

        const modelName = modelSelect.options[modelSelect.selectedIndex].text;
        mainChart.options.plugins.title.text = `${rateType} Rate (${unit}) - ${modelName}`;
        mainChart.update();
    }

    // Page Navigation
    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const targetPage = this.getAttribute('data-page');

            // Update active nav item
            document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
            this.parentElement.classList.add('active');

            // Show target page, hide others
            pages.forEach(page => {
                if (page.id === `${targetPage}-page`) {
                    page.classList.add('active');
                } else {
                    page.classList.remove('active');
                }
            });
        });
    });

    // Event Listeners
    reservoirSelect.addEventListener('change', function () {
        const reservoir = this.value;
        const rateType = rateSelect.value;

        // Update Image
        reservoirImage.src = `./assets/images/${reservoir}.png`;

        // Update Chart
        updateChart(reservoir, rateType, currentModel);

        // Update Table Data with current model
        const newData = generateTableData(reservoir, currentModel);
        allTableData = newData;
        populateTable(allTableData);
    });

    rateSelect.addEventListener('change', function () {
        const reservoir = reservoirSelect.value;
        const rateType = this.value;

        updateChart(reservoir, rateType, currentModel);
    });

    modelSelect.addEventListener('change', function () {
        currentModel = this.value;
        const reservoir = reservoirSelect.value;
        const rateType = rateSelect.value;

        console.log(`Model changed to: ${this.value}`);

        // Update chart with new model predictions (actual data stays the same)
        updateChart(reservoir, rateType, currentModel);

        // Regenerate table data with new model predictions (actual data stays the same)
        const newData = generateTableData(reservoir, currentModel);
        allTableData = newData;
        populateTable(allTableData);
    });

    // Filter Logic
    applyFiltersBtn.addEventListener('click', function () {
        const dateMinVal = filterDateMin.value;
        const dateMaxVal = filterDateMax.value;
        const actualMinVal = filterActualMin.value ? parseFloat(filterActualMin.value) : null;
        const actualMaxVal = filterActualMax.value ? parseFloat(filterActualMax.value) : null;
        const predictedMinVal = filterPredictedMin.value ? parseFloat(filterPredictedMin.value) : null;
        const predictedMaxVal = filterPredictedMax.value ? parseFloat(filterPredictedMax.value) : null;
        const statusVal = filterStatus.value;

        const filteredData = allTableData.filter(row => {
            let matchDateMin = true;
            let matchDateMax = true;
            let matchActualMin = true;
            let matchActualMax = true;
            let matchPredictedMin = true;
            let matchPredictedMax = true;
            let matchStatus = true;

            if (dateMinVal) {
                matchDateMin = row.date >= dateMinVal;
            }

            if (dateMaxVal) {
                matchDateMax = row.date <= dateMaxVal;
            }

            if (actualMinVal !== null) {
                matchActualMin = row.actual >= actualMinVal;
            }

            if (actualMaxVal !== null) {
                matchActualMax = row.actual <= actualMaxVal;
            }

            if (predictedMinVal !== null) {
                matchPredictedMin = row.predicted >= predictedMinVal;
            }

            if (predictedMaxVal !== null) {
                matchPredictedMax = row.predicted <= predictedMaxVal;
            }

            if (statusVal !== 'all') {
                matchStatus = row.status === statusVal;
            }

            return matchDateMin && matchDateMax && matchActualMin && matchActualMax &&
                matchPredictedMin && matchPredictedMax && matchStatus;
        });

        populateTable(filteredData);
    });

    resetFiltersBtn.addEventListener('click', function () {
        filterDateMin.value = '';
        filterDateMax.value = '';
        filterActualMin.value = '';
        filterActualMax.value = '';
        filterPredictedMin.value = '';
        filterPredictedMax.value = '';
        filterStatus.value = 'all';
        populateTable(allTableData);
    });

    // Initial Load
    allTableData = generateTableData('reservoir1', currentModel);
    populateTable(allTableData);
});