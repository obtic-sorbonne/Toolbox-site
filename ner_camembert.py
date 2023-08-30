import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

from transformers import pipeline
nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def ner_camembert(text, output_name):
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
