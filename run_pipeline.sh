python generic/explora_sc.py mistral7b_16 strategyqa full 12
python generic/explora_sc.py mistral7b_16 gsm8k full 12
python generic/explora_sc.py mistral7b_16 tabmwp full 12
python generic/explora_sc.py mistral7b_16 aquarat full 12
python generic/explora_sc.py mistral7b_16 finqa full 12

cp output/aquarat_mistral7b_16_selected_exemplar.csv output/aquarat_mistral7b_32_selected_exemplar.csv
cp output/strategyqa_mistral7b_16_selected_exemplar.csv output/strategyqa_mistral7b_32_selected_exemplar.csv
cp output/finqa_mistral7b_16_selected_exemplar.csv output/finqa_mistral7b_32_selected_exemplar.csv
cp output/gsm8k_mistral7b_16_selected_exemplar.csv output/gsm8k_mistral7b_32_selected_exemplar.csv
cp output/tabmwp_mistral7b_16_selected_exemplar.csv output/tabmwp_mistral7b_32_selected_exemplar.csv

cp output/aquarat_mistral7b_16_selected_exemplar.csv output/aquarat_llama3b_16_selected_exemplar.csv
cp output/strategyqa_mistral7b_16_selected_exemplar.csv output/strategyqa_llama3b_16_selected_exemplar.csv
cp output/finqa_mistral7b_16_selected_exemplar.csv output/finqa_llama3b_16_selected_exemplar.csv
cp output/gsm8k_mistral7b_16_selected_exemplar.csv output/gsm8k_llama3b_16_selected_exemplar.csv
cp output/tabmwp_mistral7b_16_selected_exemplar.csv output/tabmwp_llama3b_16_selected_exemplar.csv

cp output/aquarat_mistral7b_16_selected_exemplar.csv output/aquarat_llama1b_16_selected_exemplar.csv
cp output/strategyqa_mistral7b_16_selected_exemplar.csv output/strategyqa_llama1b_16_selected_exemplar.csv
cp output/finqa_mistral7b_16_selected_exemplar.csv output/finqa_llama1b_16_selected_exemplar.csv
cp output/gsm8k_mistral7b_16_selected_exemplar.csv output/gsm8k_llama1b_16_selected_exemplar.csv
cp output/tabmwp_mistral7b_16_selected_exemplar.csv output/tabmwp_llama1b_16_selected_exemplar.csv

cp output/aquarat_mistral7b_16_selected_exemplar.csv output/aquarat_llama3b_32_selected_exemplar.csv
cp output/strategyqa_mistral7b_16_selected_exemplar.csv output/strategyqa_llama3b_32_selected_exemplar.csv
cp output/finqa_mistral7b_16_selected_exemplar.csv output/finqa_llama3b_32_selected_exemplar.csv
cp output/gsm8k_mistral7b_16_selected_exemplar.csv output/gsm8k_llama3b_32_selected_exemplar.csv
cp output/tabmwp_mistral7b_16_selected_exemplar.csv output/tabmwp_llama3b_32_selected_exemplar.csv

cp output/aquarat_mistral7b_16_selected_exemplar.csv output/aquarat_llama1b_32_selected_exemplar.csv
cp output/strategyqa_mistral7b_16_selected_exemplar.csv output/strategyqa_llama1b_32_selected_exemplar.csv
cp output/finqa_mistral7b_16_selected_exemplar.csv output/finqa_llama1b_32_selected_exemplar.csv
cp output/gsm8k_mistral7b_16_selected_exemplar.csv output/gsm8k_llama1b_32_selected_exemplar.csv
cp output/tabmwp_mistral7b_16_selected_exemplar.csv output/tabmwp_llama1b_32_selected_exemplar.csv

python GSM8K/explora+SC.py llama1b_16 gsm8k full
python TabMwp/explora+SC_old.py llama1b_16 tabmwp full
python Strategy/Strat/strategyqa_explora_api.py llama1b_16 strategyqa full
python AquaRat/explora+SC.py llama1b_16 aquarat full
python FinQA/explora+SC.py llama1b_16 finqa full

python GSM8K/explora+SC.py llama3b_16 gsm8k full
python TabMwp/explora+SC_old.py llama3b_16 tabmwp full
python Strategy/Strat/strategyqa_explora_api.py llama3b_16 strategyqa full
python AquaRat/explora+SC.py llama3b_16 aquarat full
python FinQA/explora+SC.py llama3b_16 finqa full

python generic/explora_sc.py llama1b_16 strategyqa test 12
python generic/explora_sc.py llama1b_16 gsm8k test 12
python generic/explora_sc.py llama1b_16 tabmwp test 12
python generic/explora_sc.py llama1b_16 aquarat test 12
python generic/explora_sc.py llama1b_16 finqa test 12

python generic/explora_sc.py llama3b_16 strategyqa test 12
python generic/explora_sc.py llama3b_16 gsm8k test 12
python generic/explora_sc.py llama3b_16 tabmwp test 12
python generic/explora_sc.py llama3b_16 aquarat test 12
python generic/explora_sc.py llama3b_16 finqa test 12

python generic/explora_sc.py mistral7b_16 strategyqa test 12
python generic/explora_sc.py mistral7b_16 gsm8k test 12
python generic/explora_sc.py mistral7b_16 tabmwp test 12
python generic/explora_sc.py mistral7b_16 aquarat test 12
python generic/explora_sc.py mistral7b_16 finqa test 12