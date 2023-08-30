import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def ner_camembert(text, output_name, modele_REN):
	
	tokenizer = AutoTokenizer.from_pretrained(modele_REN)
	model = AutoModelForTokenClassification.from_pretrained(modele_REN)
	
	nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
		
	paragraphs = text.split('\n')
	text_lines = [x for x in paragraphs if x != '']
	res_ner = []
	for line in text_lines:
		res_ner += nlp(line)
		
	fields = ['word', 'entity_group', 'start', 'end', 'score']

	with open(output_name, 'w', newline='') as f_out:
		writer = csv.DictWriter(f_out, fieldnames = fields)
		writer.writeheader()
		writer.writerows(res_ner)
	
	return True
