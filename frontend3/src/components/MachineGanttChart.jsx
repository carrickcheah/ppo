import React from 'react';
import Plot from 'react-plotly.js';

const MachineGanttChart = ({ machines, timeRange = '2weeks' }) => {
  if (!machines || machines.length === 0) {
    return <div>No machine data available</div>;
  }

  // Prepare data for Plotly Gantt chart
  const prepareGanttData = () => {
    const traces = [];
    const yLabels = [];
    const shapes = [];
    let yIndex = 0;

    // Sort machines by name in DESCENDING order
    const sortedMachines = [...machines].sort((a, b) => 
      b.machine_name.localeCompare(a.machine_name)
    );

    sortedMachines.forEach(machine => {
      yLabels.push(machine.machine_name);
      
      // Add each task on this machine
      machine.tasks.forEach(task => {
        traces.push({
          x: [task.start, task.end],
          y: [yIndex, yIndex],
          mode: 'lines',
          line: {
            color: task.color,
            width: 20
          },
          name: task.task_label,
          showlegend: false,
          text: task.task_label,
          hovertemplate: 
            `<b>${task.task_label}</b><br>` +
            `Machine: ${machine.machine_name}<br>` +
            `Process: ${task.process_name}<br>` +
            `Start: ${task.start.toFixed(1)}h<br>` +
            `End: ${task.end.toFixed(1)}h<br>` +
            `Duration: ${task.duration.toFixed(1)}h<br>` +
            `Days to LCD: ${task.days_to_deadline.toFixed(1)}<br>` +
            `<extra></extra>`
        });

        // Add text annotation for job label on the bar
        if (task.duration > 20) { // Only show label if bar is wide enough
          traces.push({
            x: [(task.start + task.end) / 2],
            y: [yIndex],
            mode: 'text',
            text: [task.task_label.length > 15 ? task.task_label.substring(0, 15) + '...' : task.task_label],
            textposition: 'middle center',
            textfont: {
              size: 9,
              color: 'white',
              family: 'Arial, sans-serif'
            },
            showlegend: false,
            hoverinfo: 'skip'
          });
        }
      });
      
      yIndex++;
    });

    // Add current time line (example at 16 hours)
    const currentTime = 16 * 1;
    shapes.push({
      type: 'line',
      x0: currentTime,
      x1: currentTime,
      y0: -0.5,
      y1: yIndex - 0.5,
      line: {
        color: 'red',
        width: 2,
        dash: 'dash'
      }
    });

    return { traces, yLabels, shapes, yIndex };
  };

  const { traces, yLabels, shapes, yIndex } = prepareGanttData();

  // Calculate max time for x-axis range
  const maxTime = Math.max(
    ...machines.flatMap(m => m.tasks.map(t => t.end))
  );

  // Define x-axis range and ticks based on selected time range
  let xaxisConfig;
  if (timeRange === '2days') {
    const maxHours = 48; // 2 days in hours
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
      ticktext: ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '00:00'],
    };
  } else if (timeRange === '2weeks') {
    const maxHours = 336; // 2 weeks in hours
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336],
      ticktext: ['Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10', 'Day 11', 'Day 12', 'Day 13', 'Day 14'],
    };
  } else if (timeRange === '4weeks') {
    const maxHours = 672; // 4 weeks in hours
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: [0, 168, 336, 504, 672],
      ticktext: ['Week 0', 'Week 1', 'Week 2', 'Week 3', 'Week 4'],
    };
  } else {
    // 6 weeks
    const maxHours = 1008; // 6 weeks in hours
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: [0, 168, 336, 504, 672, 840, 1008],
      ticktext: ['Week 0', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'],
    };
  }

  const layout = {
    title: {
      text: 'Machine Allocation',
      font: { size: 20, family: 'Arial, sans-serif' },
      x: 0.5,
      xanchor: 'center'
    },
    xaxis: {
      ...xaxisConfig,
      titlefont: { size: 14 },
      showgrid: true,
      gridcolor: '#ddd',
      gridwidth: 1,
      zeroline: true,
      zerolinecolor: '#999',
      zerolinewidth: 2,
      // range is set in xaxisConfig but override if needed
      range: xaxisConfig.range || [0, maxTime + 24],
      tickfont: { size: 12 },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    yaxis: {
      title: 'Machines',
      titlefont: { size: 14 },
      showgrid: false,
      zeroline: false,
      tickmode: 'array',
      tickvals: [...Array(yIndex).keys()],
      ticktext: yLabels,
      automargin: true,
      range: [-1, yIndex],
      tickfont: { size: 11, family: 'Courier New, monospace' },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    shapes: shapes,
    autosize: true,
    height: Math.max(600, yIndex * 22 + 100),
    margin: {
      l: 150,
      r: 50,
      t: 50,
      b: 50
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    hovermode: 'closest',
    dragmode: false,
    showlegend: true,
    legend: {
      orientation: 'v',
      yanchor: 'top',
      y: 0.95,
      xanchor: 'right',
      x: 0.99,
      bgcolor: 'rgba(255,255,255,0.9)',
      bordercolor: '#ccc',
      borderwidth: 1,
      font: { size: 12 },
      title: {
        text: '<b>Status</b>',
        font: { size: 13 }
      },
      traceorder: 'normal',
      itemsizing: 'constant'
    }
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    scrollZoom: false,
    doubleClick: false,
    staticPlot: false,
    modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'machine_allocation_chart',
      width: 1400,
      scale: 1
    }
  };

  // Add custom legend traces for status colors
  const legendTraces = [
    {
      x: [null],
      y: [null],
      mode: 'markers',
      marker: { color: '#FF0000', size: 10, symbol: 'square' },
      name: 'Late (<0h)',
      showlegend: true,
      hoverinfo: 'skip'
    },
    {
      x: [null],
      y: [null],
      mode: 'markers',
      marker: { color: '#FFA500', size: 10, symbol: 'square' },
      name: 'Warning (<24h)',
      showlegend: true,
      hoverinfo: 'skip'
    },
    {
      x: [null],
      y: [null],
      mode: 'markers',
      marker: { color: '#FFFF00', size: 10, symbol: 'square' },
      name: 'Caution (<72h)',
      showlegend: true,
      hoverinfo: 'skip'
    },
    {
      x: [null],
      y: [null],
      mode: 'markers',
      marker: { color: '#00FF00', size: 10, symbol: 'square' },
      name: 'OK (>72h)',
      showlegend: true,
      hoverinfo: 'skip'
    }
  ];

  return (
    <div className="gantt-chart-container">
      <Plot
        data={[...traces, ...legendTraces]}
        layout={layout}
        config={config}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default MachineGanttChart;