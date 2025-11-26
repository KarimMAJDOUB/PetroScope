document.addEventListener('DOMContentLoaded', function () {
    const reservoirSelect = document.getElementById('reservoir-select');
    const rateSelect = document.getElementById('rate-select');
    const reservoirImage = document.getElementById('reservoir-image');
    const ctx = document.getElementById('oilRateChart').getContext('2d');
    const tableBody = document.querySelector('#data-table tbody');

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
        reservoirImage.src = `${reservoir}.png`;

        // Update Chart
        updateChart(reservoir, rateType);

        // Update Table
        const tableData = generateTableData(reservoir, 150); // Generate 150 rows
        populateTable(tableData);
    });

    rateSelect.addEventListener('change', function () {
        const reservoir = reservoirSelect.value;
        const rateType = this.value;

        updateChart(reservoir, rateType);
    });

    // Initial Load
    const initialTableData = generateTableData('reservoir1', 150);
    populateTable(initialTableData);
});
