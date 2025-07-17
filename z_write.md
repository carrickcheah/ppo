Summary what we done today. I want save into this file. So other teams members read this report, they can have right context, and follw up detail. Help them continue the unfinish works.   


Rules:
什么工人只能做什么机器（用 category,不用真人)
什么机器只能做什么工(machine)






jot = tbl_jo_txn
jop = tbl_jo_process
di  = tbl_daily_item
mac = tbl_machine
uv run python src/training/train_scaed_production.py


Question only. 

  1. Learning Rate: 0.0003 → 0.0001 (why not lower? then can learn more precisely)
  2. Entropy: 0.01 → 0.05 ( why not set to 1.0)



  3. Setup Penalty: 2.0 → 5.0
    - Makes switching products more "expensive"
    - Should reduce that 33.6% setup time
  4. Network Size: [256,256,128] → [512,512,256]
can increase brain to 1024,1024,512 ? is better to have bigger brain?


for all question above, is it practical? what is the pro and cons of each? review as








