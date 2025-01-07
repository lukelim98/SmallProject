# Import necessary libraries
import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

# Define the training data with text and labeled entities
train_data = [
    ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("What is the price of 5 laptops?", {"entities": [(21, 22, "QUANTITY"), (23, 30, "PRODUCT")]}),
    ("How much are 7 bottles?", {"entities": [(13, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("Could I buy 17 phones from you?", {"entities": [(12, 14, "QUANTITY"), (15, 21, "PRODUCT")]}),
    ("Do you have 3 apples?", {"entities": [(12, 13, "QUANTITY"), (14, 20, "PRODUCT")]}),
    ("How much for 12 oranges?", {"entities": [(12, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("Can I get 20 pencils?", {"entities": [(10, 12, "QUANTITY"), (13, 20, "PRODUCT")]}),
    ("Do you sell 8 notebooks?", {"entities": [(11, 12, "QUANTITY"), (13, 22, "PRODUCT")]}),
    ("I need 15 chairs, how much?", {"entities": [(7, 9, "QUANTITY"), (10, 16, "PRODUCT")]}),
    ("Is 1 table available?", {"entities": [(3, 4, "QUANTITY"), (5, 10, "PRODUCT")]}),
    ("Order 25 books for me.", {"entities": [(6, 8, "QUANTITY"), (9, 14, "PRODUCT")]}),
    ("Please deliver 9 boxes.", {"entities": [(15, 16, "QUANTITY"), (17, 22, "PRODUCT")]}),
    ("Can I buy 50 markers?", {"entities": [(10, 12, "QUANTITY"), (13, 20, "PRODUCT")]}),
    ("Reserve 30 tables for us.", {"entities": [(8, 10, "QUANTITY"), (11, 17, "PRODUCT")]}),
    ("Can you get 7 mugs?", {"entities": [(12, 13, "QUANTITY"), (14, 18, "PRODUCT")]}),
    ("How much is 1 lamp?", {"entities": [(13, 14, "QUANTITY"), (15, 19, "PRODUCT")]}),
    ("Are 100 chairs available?", {"entities": [(4, 7, "QUANTITY"), (8, 14, "PRODUCT")]}),
    ("Can I see 22 pens?", {"entities": [(10, 12, "QUANTITY"), (13, 17, "PRODUCT")]}),
    ("I want to buy 40 phones.", {"entities": [(15, 17, "QUANTITY"), (18, 24, "PRODUCT")]}),
    ("Do you stock 18 rugs?", {"entities": [(12, 14, "QUANTITY"), (15, 19, "PRODUCT")]}),
    ("Find me 6 mirrors.", {"entities": [(8, 9, "QUANTITY"), (10, 17, "PRODUCT")]}),
    ("How much for 14 jackets?", {"entities": [(12, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("Are 5 beds in stock?", {"entities": [(4, 5, "QUANTITY"), (6, 10, "PRODUCT")]}),
    ("Do you have 23 desks?", {"entities": [(12, 14, "QUANTITY"), (15, 20, "PRODUCT")]}),
    ("Can I get 11 glasses?", {"entities": [(10, 12, "QUANTITY"), (13, 20, "PRODUCT")]}),
    ("What is the price of 9 hats?", {"entities": [(21, 22, "QUANTITY"), (23, 27, "PRODUCT")]}),
    ("Order 2 sofas for me.", {"entities": [(6, 7, "QUANTITY"), (8, 13, "PRODUCT")]}),
    ("How much are 16 belts?", {"entities": [(13, 15, "QUANTITY"), (16, 21, "PRODUCT")]}),
    ("Sell me 30 pillows.", {"entities": [(8, 10, "QUANTITY"), (11, 18, "PRODUCT")]}),
    ("Can I buy 8 staplers?", {"entities": [(10, 11, "QUANTITY"), (12, 20, "PRODUCT")]}),
    ("Do you offer 13 plates?", {"entities": [(12, 14, "QUANTITY"), (15, 21, "PRODUCT")]}),
    ("I need 50 spoons.", {"entities": [(7, 9, "QUANTITY"), (10, 16, "PRODUCT")]}),
    ("How much for 21 baskets?", {"entities": [(12, 14, "QUANTITY"), (15, 22, "PRODUCT")]}),
    ("Can I have 33 candles?", {"entities": [(11, 13, "QUANTITY"), (14, 21, "PRODUCT")]}),
    ("Do you sell 45 bowls?", {"entities": [(11, 13, "QUANTITY"), (14, 19, "PRODUCT")]}),
    ("Can I purchase 5 hammers?", {"entities": [(15, 16, "QUANTITY"), (17, 24, "PRODUCT")]}),
    ("Get me 4 buckets.", {"entities": [(7, 8, "QUANTITY"), (9, 16, "PRODUCT")]}),
    ("Are 2 carpets in stock?", {"entities": [(4, 5, "QUANTITY"), (6, 13, "PRODUCT")]}),
    ("Buy 6 mats for me.", {"entities": [(4, 5, "QUANTITY"), (6, 10, "PRODUCT")]}),
    ("What is the price of 29 scarves?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
    ("Order 19 clocks for delivery.", {"entities": [(6, 8, "QUANTITY"), (9, 15, "PRODUCT")]})
]

# Load the pre-trained SpaCy model (medium English model)
nlp = spacy.load('en_core_web_md')

# Check if the NER pipeline is already in the model
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner') # Add a new NER pipeline component
else:
    ner = nlp.get_pipe('ner') # Retrieve the existing NER pipeline
    
# Add new entity labels to NER pipeline
for _, annotations in train_data:
    for ent in annotations['entities']:
        if ent[2] not in ner.labels:
            ner.add_label(ent[2])

# Disable other components (e.g., tagger, parser) during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()

    # Set the number of training epochs
    epochs = 50
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}

        # Create mini-batches of the training data
        batches = minibatch(train_data, size=2)
        for batch in batches:
            examples = []
            for text, annotations in batch:
                # Create a SpaCy Example object from text and annotations
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            # Update the NER model with the exampels and compute losses
            nlp.update(examples, drop=0.5, losses=losses)

        # Print losses after each epoch to monitor training epochs
        print(f"Epoch {epoch+1}, Losses: {losses}")

# Save the trained NER model to disk for later use
nlp.to_disk('custom_ner_model')

# Load the trained NER model
trained_nlp = spacy.load('custom_ner_model')

# Define test texts to evalute the trained model
test_texts = [
    "How much for 3 oranges?",
    "I want 15 chairs for the conference.",
    "Can you give me the price for 6 desks?",
]

# Use the trained model to make predictiosn on test texts
for text in test_texts:
    doc = trained_nlp(text)
    print(f"Text: {text}")
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print()
