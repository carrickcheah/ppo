// Debug script to check localStorage data
const cacheData = localStorage.getItem('aiOptimizerCache');
if (cacheData) {
  const parsed = JSON.parse(cacheData);
  console.log('=== DEBUG: Cached Data Analysis ===');
  console.log('Total ganttPriorityView items:', parsed.ganttPriorityView?.length);
  
  if (parsed.ganttPriorityView?.length > 0) {
    console.log('\nFirst 5 Task IDs:');
    parsed.ganttPriorityView.slice(0, 5).forEach((item, idx) => {
      console.log(`${idx + 1}. Task: "${item.Task}"`);
      console.log(`   JobFamily: "${item.JobFamily}"`);
      console.log(`   Resource: "${item.Resource}"`);
    });
    
    // Check if Task contains process notation
    const hasProcessNotation = parsed.ganttPriorityView.some(item => 
      item.Task && item.Task.includes('_') && item.Task.includes('-') && item.Task.includes('/')
    );
    console.log('\nTasks have process notation (_XXX-X/X format):', hasProcessNotation);
    
    // Check unique Task values
    const uniqueTasks = [...new Set(parsed.ganttPriorityView.map(item => item.Task))];
    console.log('\nUnique Task count:', uniqueTasks.length);
    console.log('Sample unique tasks:', uniqueTasks.slice(0, 5));
  }
} else {
  console.log('No cache data found in localStorage');
}