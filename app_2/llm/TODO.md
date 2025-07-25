# LLM Scheduler Implementation TODO

## Phase 1: Foundation (Day 1-2)

### Setup & Configuration
- [x] Create directory structure /app_2/llm/
- [x] Initialize Python package with __init__.py
- [ ] Create requirements.txt with dependencies
- [ ] Setup .env loading for DeepSeek API key
- [ ] Create config.yaml for LLM settings

### DeepSeek Client Implementation
- [ ] Create deepseek_client.py with OpenAI compatibility
- [ ] Implement authentication using API key
- [ ] Add retry logic with exponential backoff
- [ ] Create error handling for API failures
- [ ] Add support for streaming responses
- [ ] Implement token counting and cost tracking

### Data Integration
- [ ] Create data_adapter.py for format conversion
- [ ] Load snapshot files from phase3/snapshots/
- [ ] Convert job data to LLM-friendly text format
- [ ] Convert machine data to readable format
- [ ] Implement schedule output formatter
- [ ] Add compatibility with existing API format

## Phase 2: Core Scheduling (Day 3-4)

### Prompt Engineering
- [ ] Create prompts.py with template system
- [ ] Design basic scheduling prompt
- [ ] Add chain-of-thought reasoning prompt
- [ ] Create few-shot examples
- [ ] Design constraint-aware prompts
- [ ] Add multi-objective optimization prompts

### LLM Scheduler Core
- [ ] Implement llm_scheduler.py main class
- [ ] Create schedule() method
- [ ] Implement prompt generation logic
- [ ] Add response parsing
- [ ] Handle multi-machine job prompts
- [ ] Implement iterative refinement

### Response Processing
- [ ] Create parser.py for text parsing
- [ ] Parse job assignments
- [ ] Extract start/end times
- [ ] Handle multiple output formats
- [ ] Add error recovery for malformed responses
- [ ] Create structured schedule objects

## Phase 3: Validation & Constraints (Day 5)

### Constraint Validation
- [ ] Create validator.py
- [ ] Implement sequence constraint checking
- [ ] Add machine compatibility validation
- [ ] Check for time overlaps
- [ ] Validate multi-machine requirements
- [ ] Add working hours validation (optional)

### Error Handling
- [ ] Handle constraint violations
- [ ] Implement schedule repair logic
- [ ] Add fallback strategies
- [ ] Create detailed error messages
- [ ] Log validation failures

## Phase 4: Evaluation & Testing (Day 6-7)

### Evaluation Framework
- [ ] Create evaluator.py
- [ ] Implement makespan calculation
- [ ] Add on-time delivery rate
- [ ] Calculate machine utilization
- [ ] Compare with baseline methods
- [ ] Generate performance reports

### Batch Processing
- [ ] Create batch_runner.py
- [ ] Test multiple scenarios
- [ ] Run different prompt strategies
- [ ] Collect statistics
- [ ] Create comparison charts

### Testing
- [ ] Write unit tests for all components
- [ ] Create integration tests
- [ ] Test with edge cases
- [ ] Performance benchmarks
- [ ] Load testing with 500+ jobs

## Phase 5: API & Deployment (Day 8)

### FastAPI Integration
- [ ] Create api_server.py
- [ ] Implement POST /llm/schedule endpoint
- [ ] Add request/response models
- [ ] Handle async processing
- [ ] Add error handling
- [ ] Create health check endpoint

### Documentation
- [ ] Write comprehensive README.md
- [ ] Create API documentation
- [ ] Add usage examples
- [ ] Document performance metrics
- [ ] Create troubleshooting guide

### Examples
- [ ] Simple 10-job scheduling example
- [ ] Complex multi-machine example
- [ ] Rush order scenario
- [ ] Comparison with PPO results
- [ ] Performance analysis notebook

## Performance Targets
- [ ] Handle 100 jobs in < 10 seconds
- [ ] Achieve 90%+ constraint satisfaction
- [ ] 80%+ on-time delivery rate
- [ ] Cost < $0.10 per schedule
- [ ] Support batch processing

## Future Enhancements
- [ ] Implement caching for repeated patterns
- [ ] Add parallel processing for large batches
- [ ] Create hybrid LLM-optimization approach
- [ ] Add real-time schedule updates
- [ ] Implement continuous learning from feedback

## Current Status
- **Started**: 2025-07-25
- **Phase**: 1 - Foundation
- **Next Steps**: Setup DeepSeek client and data integration