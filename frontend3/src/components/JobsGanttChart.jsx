import React from 'react';
import Plot from 'react-plotly.js';

const JobsGanttChart = ({ jobs }) => {
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

    // Sort job groups for consistent display
    const sortedKeys = Object.keys(jobGroups).sort((a, b) => {
      // Extract family and sequence for proper sorting
      const jobA = jobGroups[a][0];
      const jobB = jobGroups[b][0];
      
      // Sort by family ID first
      if (jobA.job_id !== jobB.job_id) {
        return jobA.job_id.localeCompare(jobB.job_id);
      }
      // Then by sequence number
      return jobA.sequence - jobB.sequence;
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

  const layout = {
    title: {
      text: 'Production Planning System - Job Allocation',
      font: { size: 20, family: 'Arial, sans-serif' },
      x: 0.5,
      xanchor: 'center'
    },
    xaxis: {
      title: 'Time (Hours)',
      titlefont: { size: 14 },
      showgrid: true,
      gridcolor: '#ddd',
      gridwidth: 1,
      zeroline: true,
      zerolinecolor: '#999',
      zerolinewidth: 2,
      range: [0, Math.max(...jobs.map(j => j.end)) * 1.1],
      tickmode: 'linear',
      tick0: 0,
      dtick: 200,
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
    height: Math.max(800, yIndex * 25 + 200),
    margin: {
      l: 280,
      r: 150,
      t: 100,
      b: 100
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    hovermode: 'closest',
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
    <div className="gantt-chart-container" style={{ width: '100%', height: '100%' }}>
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