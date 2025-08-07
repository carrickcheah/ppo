import React from 'react';
import Plot from 'react-plotly.js';

const JobsGanttChart = ({ jobs, timeRange = '2weeks' }) => {
  if (!jobs || jobs.length === 0) {
    return <div>No job data available</div>;
  }

  // Prepare data for Plotly Gantt chart
  const prepareGanttData = () => {
    // Group jobs by family and sequence for proper row ordering
    const jobGroups = {};
    
    jobs.forEach(job => {
      const key = job.task_label;
      if (!jobGroups[key]) {
        jobGroups[key] = [];
      }
      jobGroups[key].push(job);
    });

    // Create traces for each job
    const traces = [];
    const yLabels = [];
    const shapes = [];
    let yIndex = 0;

    // Sort job groups for consistent display - DESCENDING order
    const sortedKeys = Object.keys(jobGroups).sort((a, b) => {
      // Extract family and sequence for proper sorting
      const jobA = jobGroups[a][0];
      const jobB = jobGroups[b][0];
      
      // Sort by family ID first (descending)
      if (jobA.job_id !== jobB.job_id) {
        return jobB.job_id.localeCompare(jobA.job_id);
      }
      // Then by sequence number (descending)
      return jobB.sequence - jobA.sequence;
    });

    sortedKeys.forEach(key => {
      const jobList = jobGroups[key];
      
      jobList.forEach(job => {
        // Use clean, simple labels
        yLabels.push(job.task_label);
        
        traces.push({
          x: [job.start, job.end],
          y: [yIndex, yIndex],
          mode: 'lines',
          line: {
            color: job.color,
            width: 18
          },
          name: job.task_label,
          showlegend: false,
          hovertemplate: 
            `<b>${job.task_label}</b><br>` +
            `Process: ${job.process_name}<br>` +
            `Machine: ${job.machine}<br>` +
            `Start: ${job.start.toFixed(1)}h<br>` +
            `End: ${job.end.toFixed(1)}h<br>` +
            `Duration: ${job.duration.toFixed(1)}h<br>` +
            `Days to LCD: ${job.days_to_deadline.toFixed(1)}<br>` +
            `<extra></extra>`
        });
        
        yIndex++;
      });
    });

    // Add current time line (example at 16 hours - 16:00)
    const currentTime = 16 * 1; // Current time in hours
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
      text: 'Production Planning System - Job Allocation',
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
      range: xaxisConfig.range || [0, Math.max(...jobs.map(j => j.end)) + 24],
      tickfont: { size: 12 },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    yaxis: {
      title: 'Jobs',
      titlefont: { size: 14 },
      showgrid: false,
      zeroline: false,
      tickmode: 'array',
      tickvals: [...Array(yIndex).keys()],
      ticktext: yLabels,
      automargin: true,
      range: [-1, yIndex],
      tickfont: { size: 10, family: 'Arial, sans-serif' },
      showline: true,
      linecolor: '#999',
      linewidth: 2
    },
    shapes: shapes,
    autosize: true,
    height: Math.max(600, yIndex * 22 + 100),
    margin: {
      l: 180,
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
      filename: 'job_allocation_chart',
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

export default JobsGanttChart;