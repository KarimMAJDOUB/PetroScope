document.addEventListener('DOMContentLoaded', function () {
    const reservoirSelect = document.getElementById('reservoir-select');
    const rateSelect = document.getElementById('rate-select');
    const modelSelect = document.getElementById('model-select');
    const reservoirImage = document.getElementById('reservoir-image');
    const ctx = document.getElementById('oilRateChart').getContext('2d');
    const tableBody = document.querySelector('#data-table tbody');

    // Filter Elements
    const filterDate = document.getElementById('filter-date');
    const filterReservoir = document.getElementById('filter-reservoir');
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

    // Generate Dummy Data for Table
    function generateTableData(reservoir, count = 100) {
        const data = [];
        const baseRate = reservoir === 'reservoir1' ? 1200 : (reservoir === 'reservoir2' ? 900 : 1500);
        const startDate = new Date('2023-01-01');

        for (let i = 0; i < count; i++) {
            const currentDate = new Date(startDate);
            currentDate.setDate(startDate.getDate() + i);

            const actual = Math.floor(baseRate + (Math.random() - 0.5) * 100);
            const predicted = Math.floor(baseRate + (Math.random() - 0.5) * 50);
            const status = Math.abs(actual - predicted) > 80 ? 'Check' : 'Normal';

            data.push({
                date: currentDate.toISOString().split('T')[0],
                reservoir: reservoir.charAt(0).toUpperCase() + reservoir.slice(1),
                actual: actual,
                predicted: predicted,
                status: status
            });
        }
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
    function updateChart(reservoir, rateType) {
        // Simulate data based on inputs
        let baseValue = 1000;
        if (reservoir === 'reservoir2') baseValue = 800;
        if (reservoir === 'reservoir3') baseValue = 1400;

        if (rateType === 'Gas') baseValue = baseValue * 5; // Gas rates are higher numbers usually (mcf)
        if (rateType === 'Water') baseValue = baseValue * 0.5;

        const newDataActual = Array.from({ length: 6 }, () => Math.floor(baseValue + (Math.random() - 0.5) * (baseValue * 0.1)));
        const newDataPredicted = Array.from({ length: 6 }, () => Math.floor(baseValue + (Math.random() - 0.5) * (baseValue * 0.05)));

        mainChart.data.datasets[0].data = newDataActual;
        mainChart.data.datasets[1].data = newDataPredicted;

        let unit = 'bbl/d';
        if (rateType === 'Gas') unit = 'mcf/d';

        mainChart.options.plugins.title.text = `${rateType} Rate (${unit})`;
        mainChart.update();
    }

    // Event Listeners
    reservoirSelect.addEventListener('change', function () {
        const reservoir = this.value;
        const rateType = rateSelect.value;

        // Update Image
        reservoirImage.src = `./assets/images/${reservoir}.png`;

        // Update Chart
        updateChart(reservoir, rateType);

        // Update Table Data (regenerate for simplicity or filter if we had a real backend)
        // For this demo, we'll regenerate data for the selected reservoir to simulate a "view" change
        // But we also want to keep "allTableData" populated for the global filter if needed.
        // Let's say the main view follows the reservoir select, but the filter can override.

        const newData = generateTableData(reservoir, 150);
        allTableData = newData; // Update current dataset
        populateTable(allTableData);
    });

    rateSelect.addEventListener('change', function () {
        const reservoir = reservoirSelect.value;
        const rateType = this.value;

        updateChart(reservoir, rateType);
    });

    modelSelect.addEventListener('change', function () {
        console.log(`Model changed to: ${this.value}`);
        // Placeholder for model change logic
        // Maybe update chart title to indicate model used?
        mainChart.options.plugins.title.text += ` [${this.options[this.selectedIndex].text}]`;
        mainChart.update();
    });

    // Filter Logic
    applyFiltersBtn.addEventListener('click', function () {
        const dateVal = filterDate.value;
        const reservoirVal = filterReservoir.value;
        const statusVal = filterStatus.value;

        const filteredData = allTableData.filter(row => {
            let matchDate = true;
            let matchReservoir = true;
            let matchStatus = true;

            if (dateVal) {
                matchDate = row.date === dateVal;
            }

            if (reservoirVal !== 'all') {
                // row.reservoir is "Reservoir 1", value is "reservoir1"
                // Need to normalize or map.
                // row.reservoir format: "Reservoir 1"
                // reservoirVal format: "reservoir1"
                // Let's just check if row.reservoir includes the number or something simple
                // Or construct the expected string
                const expectedResString = reservoirVal.replace('reservoir', 'Reservoir ');
                matchReservoir = row.reservoir === expectedResString;
            }

            if (statusVal !== 'all') {
                matchStatus = row.status === statusVal;
            }

            return matchDate && matchReservoir && matchStatus;
        });

        populateTable(filteredData);
    });

    resetFiltersBtn.addEventListener('click', function () {
        filterDate.value = '';
        filterReservoir.value = 'all';
        filterStatus.value = 'all';
        populateTable(allTableData);
    });

    // Initial Load
    allTableData = generateTableData('reservoir1', 150);
    populateTable(allTableData);
});