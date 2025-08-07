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
            width: 30
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
              size: 10,
              color: 'black',
              family: 'Arial, sans-serif',
              weight: 'bold'
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
  if (timeRange === '5days') {
    const maxHours = 120; // 5 days in hours
    // Generate hourly ticks every 12 hours with 24-hour format
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i <= maxHours; i += 12) {
      tickvals.push(i);
      const hour = i % 24;
      const hourStr = hour.toString().padStart(2, '0');
      ticktext.push(`${hourStr}:00`);
    }
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: tickvals,
      ticktext: ticktext,
    };
  } else if (timeRange === '2days') {
    const maxHours = 48; // 2 days in hours
    // Generate hourly ticks every 4 hours with 24-hour format
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i <= maxHours; i += 4) {
      tickvals.push(i);
      const hour = i % 24;
      const hourStr = hour.toString().padStart(2, '0');
      ticktext.push(`${hourStr}:00`);
    }
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: tickvals,
      ticktext: ticktext,
    };
  } else if (timeRange === '2weeks') {
    const maxHours = 336; // 2 weeks in hours
    // Generate hourly ticks every 24 hours with 24-hour format
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i <= maxHours; i += 24) {
      tickvals.push(i);
      const hour = i % 24;
      const hourStr = hour.toString().padStart(2, '0');
      ticktext.push(`${hourStr}:00`);
    }
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: tickvals,
      ticktext: ticktext,
    };
  } else if (timeRange === '4weeks') {
    const maxHours = 672; // 4 weeks in hours
    // Generate hourly ticks every 24 hours
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i <= maxHours; i += 24) {
      tickvals.push(i);
      ticktext.push('00:00');
    }
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: tickvals,
      ticktext: ticktext,
    };
  } else {
    // 6 weeks
    const maxHours = 1008; // 6 weeks in hours
    // Generate hourly ticks every 48 hours
    const tickvals = [];
    const ticktext = [];
    for (let i = 0; i <= maxHours; i += 48) {
      tickvals.push(i);
      ticktext.push('00:00');
    }
    xaxisConfig = {
      title: '',
      range: [0, maxHours],
      tickmode: 'array',
      tickvals: tickvals,
      ticktext: ticktext,
    };
  }

  const layout = {
    title: {
      text: '',
      font: { size: 20, family: 'Arial, sans-serif' },
      x: 0.5,
      xanchor: 'center'
    },
    xaxis: {
      ...xaxisConfig,
      titlefont: { size: 14, color: 'black' },
      showgrid: true,
      gridcolor: '#e5e7eb',
      gridwidth: 1,
      zeroline: true,
      zerolinecolor: '#999',
      zerolinewidth: 2,
      // range is set in xaxisConfig but override if needed
      range: xaxisConfig.range || [0, maxTime + 24],
      tickfont: { size: 11, color: 'black' },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    yaxis: {
      title: 'Machines',
      titlefont: { size: 14, color: 'black' },
      showgrid: false,
      zeroline: false,
      tickmode: 'array',
      tickvals: [...Array(yIndex).keys()],
      ticktext: yLabels,
      automargin: true,
      range: [-1, yIndex],
      tickfont: { size: 11, family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', color: 'black' },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    shapes: shapes,
    autosize: true,
    height: yIndex * 50 + 400,
    margin: {
      l: 280,
      r: 100,
      t: 40,
      b: 150
    },
    plot_bgcolor: '#fafbfc',
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
    <div className="gantt-chart-container" style={{ width: '100%', height: `${yIndex * 50 + 400}px` }}>
      <Plot
        data={[...traces, ...legendTraces]}
        layout={layout}
        config={config}
        useResizeHandler={false}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default MachineGanttChart;