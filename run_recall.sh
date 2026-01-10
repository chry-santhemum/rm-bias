python train.py \
--student_model recall-affirm \
--teacher_model recall-anti-affirm \
--dataset handpick \
--topic_ids 0 \
--num_seeds 10 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 10 \
--train_batch_sizes 32 \
--m_var 0 \
--n_planner_requests 32 \
--val_split_size 0 \
--context vanilla


python train.py \
--student_model recall-headers \
--teacher_model recall-anti-headers \
--dataset handpick \
--topic_ids 3 \
--num_seeds 10 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 10 \
--train_batch_sizes 32 \
--m_var 0 \
--n_planner_requests 32 \
--val_split_size 0 \
--context vanilla


python train.py \
--student_model recall-list \
--teacher_model recall-anti-list \
--dataset handpick \
--topic_ids 10 \
--num_seeds 10 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 10 \
--train_batch_sizes 32 \
--m_var 0 \
--n_planner_requests 32 \
--val_split_size 0 \
--context vanilla
