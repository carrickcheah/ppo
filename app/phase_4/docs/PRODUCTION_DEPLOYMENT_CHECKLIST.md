# Production Deployment Checklist

## Pre-Deployment Requirements

### Infrastructure
- [ ] **Server specifications met**
  - [ ] CPU: 4+ cores available
  - [ ] RAM: 16GB minimum
  - [ ] Storage: 10GB free space
  - [ ] Network: Stable connection to MariaDB

- [ ] **Software prerequisites**
  - [ ] Python 3.12+ installed
  - [ ] UV package manager installed
  - [ ] MariaDB client libraries
  - [ ] System monitoring tools

### Security
- [ ] **Access controls configured**
  - [ ] API key generated and secured
  - [ ] Database credentials encrypted
  - [ ] Firewall rules configured
  - [ ] SSL certificates ready (if HTTPS)

- [ ] **Audit requirements**
  - [ ] Logging directory created with proper permissions
  - [ ] Log rotation configured
  - [ ] Audit trail mechanism tested
  - [ ] Compliance requirements reviewed

### Data Preparation
- [ ] **Production data validated**
  - [ ] Run data ingestion: `uv run python src/data_ingestion/ingest_data.py`
  - [ ] Verify snapshot created: `ls -la data/real_production_snapshot.json`
  - [ ] Check data integrity: `uv run python scripts/verify_production_data.py`
  - [ ] Confirm job count matches expectations

- [ ] **Model verification**
  - [ ] Model file exists: `models/full_production/final_model.zip`
  - [ ] Model version matches code version
  - [ ] Test model loading: `uv run python scripts/verify_model.py`
  - [ ] Backup model created

## Deployment Steps

### Phase 1: Environment Setup (Day 1)
- [ ] **Repository deployment**
  - [ ] Clone repository to production server
  - [ ] Checkout correct version/tag
  - [ ] Verify all files present
  - [ ] Set appropriate file permissions

- [ ] **Dependencies installation**
  - [ ] Create virtual environment: `uv venv`
  - [ ] Install dependencies: `uv sync`
  - [ ] Verify installations: `uv run python -c "import stable_baselines3"` 
  - [ ] Test database connection

- [ ] **Configuration**
  - [ ] Create `.env` file from template
  - [ ] Set all environment variables
  - [ ] Configure logging paths
  - [ ] Set appropriate batch size (default: 170)

### Phase 2: Integration Testing (Day 2)
- [ ] **Component testing**
  - [ ] Test data ingestion pipeline
  - [ ] Test model loading
  - [ ] Test environment initialization
  - [ ] Test API endpoints

- [ ] **End-to-end testing**
  - [ ] Run: `uv run python phase_4/test_end_to_end_production.py`
  - [ ] Verify all tests pass
  - [ ] Check performance metrics
  - [ ] Review generated schedules

- [ ] **Load testing**
  - [ ] Test with expected production load
  - [ ] Monitor resource usage
  - [ ] Verify response times < 1s
  - [ ] Check memory stability

### Phase 3: Shadow Mode (Days 3-7)
- [ ] **Parallel deployment**
  - [ ] Deploy PPO scheduler alongside existing system
  - [ ] Configure to receive duplicate requests
  - [ ] Log all schedules without using them
  - [ ] No impact on production

- [ ] **Comparison analysis**
  - [ ] Compare PPO schedules with current system
  - [ ] Analyze makespan differences
  - [ ] Check constraint compliance
  - [ ] Document any anomalies

- [ ] **Performance monitoring**
  - [ ] Track API response times
  - [ ] Monitor resource usage patterns
  - [ ] Check batch processing frequency
  - [ ] Verify completion rates = 100%

### Phase 4: Gradual Rollout (Days 8-14)
- [ ] **10% traffic (Days 8-9)**
  - [ ] Route 10% of scheduling requests to PPO
  - [ ] Monitor closely for issues
  - [ ] Keep fallback ready
  - [ ] Collect operator feedback

- [ ] **25% traffic (Days 10-11)**
  - [ ] Increase to 25% if stable
  - [ ] Continue monitoring
  - [ ] Analyze scheduling quality
  - [ ] Check system stability

- [ ] **50% traffic (Days 12-13)**
  - [ ] Increase to 50% if successful
  - [ ] Full performance analysis
  - [ ] Operator training if needed
  - [ ] Prepare for full deployment

- [ ] **Decision checkpoint (Day 14)**
  - [ ] Review all metrics
  - [ ] Get stakeholder approval
  - [ ] Plan full cutover
  - [ ] Update documentation

### Phase 5: Full Production (Day 15+)
- [ ] **Complete cutover**
  - [ ] Route 100% traffic to PPO scheduler
  - [ ] Keep legacy system on standby
  - [ ] Monitor intensively first 24 hours
  - [ ] Have rollback plan ready

- [ ] **Stabilization**
  - [ ] Daily monitoring for first week
  - [ ] Address any issues immediately
  - [ ] Collect comprehensive metrics
  - [ ] Fine-tune if necessary

## Monitoring Checklist

### Real-time Monitoring
- [ ] **System health**
  - [ ] API up/down status
  - [ ] Response time tracking
  - [ ] Error rate monitoring
  - [ ] Resource utilization

- [ ] **Business metrics**
  - [ ] Jobs scheduled per hour
  - [ ] Average makespan
  - [ ] Completion rate
  - [ ] Batch processing events

### Alerting Configuration
- [ ] **Critical alerts**
  - [ ] API down
  - [ ] Completion rate < 95%
  - [ ] Response time > 5s
  - [ ] Memory usage > 90%

- [ ] **Warning alerts**
  - [ ] High batch frequency
  - [ ] Increasing error rate
  - [ ] Unusual scheduling patterns
  - [ ] Database connection issues

## Rollback Plan

### Immediate Rollback Triggers
- [ ] Completion rate drops below 90%
- [ ] Critical errors in production
- [ ] Response time consistently > 10s
- [ ] System instability

### Rollback Procedure
1. [ ] Stop routing new requests to PPO
2. [ ] Switch back to legacy system
3. [ ] Preserve all logs and data
4. [ ] Notify stakeholders
5. [ ] Begin root cause analysis

## Post-Deployment

### Week 1
- [ ] Daily standup on system performance
- [ ] Review all operator feedback
- [ ] Analyze scheduling improvements
- [ ] Document any issues

### Week 2-4  
- [ ] Weekly performance review
- [ ] Optimization opportunities
- [ ] Plan for Phase 5 improvements
- [ ] Knowledge transfer sessions

### Month 1 Review
- [ ] Comprehensive performance analysis
- [ ] ROI calculation
- [ ] Lessons learned documentation
- [ ] Future roadmap planning

## Success Criteria

### Technical Success
- [ ] 100% job completion rate
- [ ] API response time < 1 second
- [ ] System stability > 99.9%
- [ ] No critical production issues

### Business Success  
- [ ] Improved makespan vs legacy
- [ ] Reduced scheduling time
- [ ] Positive operator feedback
- [ ] Measurable efficiency gains

## Sign-offs

### Technical Approval
- [ ] Development team lead
- [ ] Operations manager
- [ ] Database administrator
- [ ] Security officer

### Business Approval
- [ ] Production manager
- [ ] Plant manager
- [ ] IT director
- [ ] Executive sponsor

---

**Deployment Date**: _________________

**Deployment Team**: _________________

**Emergency Contacts**: _________________