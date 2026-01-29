ifeq ($(OS),Windows_NT)
    SEP = \\
else
    SEP = /
endif

include .env

clean-python:
	poetry run isort .
	poetry run black .

clean-study1-codebook:
	poetry run python -c "from app.ml.clean import clean_study1_codebook; clean_study1_codebook()"

clean-study2-codebook:
	poetry run python -c "from app.ml.clean import clean_study2_codebook; clean_study2_codebook()"

clean-merge-features: 
	poetry run python -c "from app.ml.clean import clean_merge_features; clean_merge_features()"

create-features:
	poetry run python -c "from app.ml.feature_engineering import feature_engineering; feature_engineering()"

score-features:
	poetry run python -c "from app.ml.feature_selection import feature_selection; feature_selection()"

build-sandbox:
	poetry run python app$(SEP)ml$(SEP)sandbox.py

generate-example:
	poetry run python -c "from app.ml.deploy import deploy; deploy('generate_questionnaire_and_example')"

predict-for-example:
	poetry run python -c "from app.ml.deploy import deploy; deploy('predict_for_example')"

run-crossval:
	poetry run python -c "from app.ml.crossval import run_crossval; run_crossval()"

eda:
	poetry run python -c "from app.ml.eda import score_eda; score_eda()"

launch-visuals:
	poetry run python -c "import os, sys; project_root = os.path.abspath(os.getcwd()); os.chdir(project_root); from streamlit.web.cli import main; sys.argv = ['streamlit', 'run', 'app/ml/visuals.py', '--client.showSidebarNavigation=False']; main()"

train-best-model:
	poetry run python -c "from app.ml.deploy import deploy; deploy('train_best_model')"

assign-conditions:
	poetry run python -c "from scripts.assign_conditions import assign_conditions; assign_conditions()"

cloud-connect:
	gcloud compute ssh datagotchi

create-model:
	poetry run python -c "from scripts.train_recommender_model import create_recommender_model; create_recommender_model()"

display-users:
	poetry run python -c "from scripts.view_databases import display_users; display_users()"

display-logs:
	poetry run python -c "from scripts.view_databases import display_logs; display_logs()"

display-questions:
	poetry run python -c "from scripts.view_databases import display_questions; display_questions()"

display-answers:
	poetry run python -c "from scripts.view_databases import display_answers; display_answers()"

display-activity:
	poetry run python -c "from scripts.view_databases import display_activity; display_activity()"

reload-experiment:
	poetry run python -c "from scripts.reload_experiment import reload_databases; reload_databases()"

reload-experiment-vm:
	poetry run python -c "from scripts.reload_experiment_vm import reload_databases; reload_databases()"

update-deploy:
	git pull
	sudo supervisorctl stop microapp
	flask db upgrade
	sudo supervisorctl start microapp

sup-start:
	sudo supervisorctl start microapp

sup-stop:
	sudo supervisorctl stop microapp

start-website:
	poetry run python microapp.py --config=default

debug:
	poetry run python microapp.py --config=debug

send-files:
	gcloud compute scp --recurse $(DATA_WEBAPP_PATH) $(vm):datagotchi_sante/deploy/data

download-files:
	gcloud compute scp --recurse $(vm):datagotchi_sante/deploy/data/experiment $(DATA_EXPERIMENT_PATH) 

download-track:
	gcloud compute scp --recurse $(vm):datagotchi_sante/deploy/data/track $(DATA_EXPERIMENT_PATH) 

dump-database:
	poetry run python -c "from scripts.export_db import export_database_to_csv; export_database_to_csv()"

track-database:
	poetry run python -c "from scripts.export_db import export_database_to_csv; export_database_to_csv('deploy/data/track')"

regular-track:
	poetry run python scripts/regular_runner.py "make track-database"

regular-download:
	poetry run python scripts/regular_runner.py \
		"make download-track vm=$(vm) DATA_EXPERIMENT_PATH='$(DATA_EXPERIMENT_PATH)'"

print-feature-contrib:
	poetry run python -c "from app.ml.plots import print_feature_contribution_table; print_feature_contribution_table()"

print-vif:
	poetry run python -c "from app.ml.plots import print_vif; print_vif()"

print-feature-weights:
	poetry run python -c "from app.ml.plots import print_feature_weights; print_feature_weights()"

print-boxplots:
	poetry run python -c "from app.ml.plots import print_boxplots; print_boxplots()"

print-sociodemos:
	poetry run python -c "from app.ml.plots import print_post_exclusion_sociodemos; print_post_exclusion_sociodemos()"

print-missing-data:
	poetry run python -c "from app.ml.plots import print_missing_data_rates; print_missing_data_rates()"

print-target-distribution:
	poetry run python -c "from app.ml.plots import print_target_distribution; print_target_distribution()"

update-prolific:
	@if [ -z "$(STUDY_ID)" ] || [ -z "$(COMPLETION_CODE)" ]; then \
		echo "Usage: make update-prolific STUDY_ID=<id> COMPLETION_CODE=<code>"; \
		exit 1; \
	fi
	sed -i -e "s/^PROLIFIC_STUDY_ID=.*/PROLIFIC_STUDY_ID=$(STUDY_ID)/" -e "s/^PROLIFIC_COMPLETION_CODE=.*/PROLIFIC_COMPLETION_CODE=$(COMPLETION_CODE)/" .env
	@echo "Updated PROLIFIC_STUDY_ID and PROLIFIC_COMPLETION_CODE in .env"

visualize-logs:
	poetry run streamlit run scripts/streamlit_logs.py

filter-results:
	poetry run python -c "from scripts.results_analyses import write_filtered_db; write_filtered_db()"

clean-results:
	poetry run python -c "from scripts.results_analyses import clean_results; clean_results()"
