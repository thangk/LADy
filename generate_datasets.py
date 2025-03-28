import os
import random
import xml.dom.minidom as md
from tqdm import tqdm

# Define aspects and categories for restaurant reviews
ASPECTS = {
    "food": ["food", "meal", "dish", "menu", "cuisine", "breakfast", "lunch", "dinner", "appetizer", "dessert", "taste", "flavors", "ingredients", "portion", "pizza", "burger", "pasta", "steak", "seafood", "salad"],
    "service": ["service", "staff", "server", "waiter", "waitress", "hostess", "manager", "customer service", "reservation", "attention", "courtesy", "hospitality"],
    "ambience": ["ambience", "atmosphere", "decor", "interior", "music", "lighting", "noise", "seating", "space", "view", "comfort", "design", "environment", "vibe"],
    "price": ["price", "value", "cost", "bill", "check", "money", "worth", "affordable", "expensive", "cheap", "budget", "discount", "deal"],
    "location": ["location", "parking", "accessibility", "neighborhood", "distance", "convenience", "address", "street", "area"]
}

# Define sentiment expressions
POSITIVE_EXPRESSIONS = ["excellent", "amazing", "outstanding", "fantastic", "delicious", "wonderful", "great", "good", "lovely", "impressive", "perfect", "enjoyable", "friendly", "helpful", "attentive", "professional", "beautiful", "cozy", "romantic", "relaxing", "reasonable", "fair", "convenient", "well-located"]
NEGATIVE_EXPRESSIONS = ["terrible", "awful", "disappointing", "bad", "poor", "mediocre", "bland", "tasteless", "rude", "unhelpful", "inattentive", "unprofessional", "slow", "noisy", "uncomfortable", "crowded", "dirty", "overpriced", "expensive", "inconvenient", "poorly-located"]
NEUTRAL_EXPRESSIONS = ["average", "okay", "decent", "standard", "normal", "typical", "basic", "simple", "moderate", "plain", "regular"]

# Templates for explicit aspect mentions
EXPLICIT_TEMPLATES = {
    "food": [
        "The {aspect} was {sentiment}.",
        "I found the {aspect} to be {sentiment}.",
        "Their {aspect} is {sentiment}.",
        "{aspect} was {sentiment} and worth trying.",
        "The restaurant's {aspect} was {sentiment}.",
        "I ordered the {aspect} which was {sentiment}.",
        "We tried their {aspect} and it was {sentiment}.",
        "The {aspect} they serve is {sentiment}.",
        "Their signature {aspect} was {sentiment}.",
        "I've never had such {sentiment} {aspect} before."
    ],
    "service": [
        "The {aspect} was {sentiment}.",
        "I found the {aspect} to be {sentiment}.",
        "Their {aspect} is {sentiment}.",
        "The {aspect} we received was {sentiment}.",
        "We were greeted with {sentiment} {aspect}.",
        "The {aspect} provided was {sentiment}.",
        "They offer {sentiment} {aspect}.",
        "Their {aspect} team was {sentiment}.",
        "The restaurant's {aspect} was {sentiment}.",
        "I was impressed by their {sentiment} {aspect}."
    ],
    "ambience": [
        "The {aspect} was {sentiment}.",
        "I found the {aspect} to be {sentiment}.",
        "Their {aspect} is {sentiment}.",
        "The restaurant's {aspect} was {sentiment}.",
        "We enjoyed the {sentiment} {aspect}.",
        "The {aspect} made us feel {sentiment}.",
        "They've created a {sentiment} {aspect}.",
        "The {aspect} of the place is {sentiment}.",
        "We dined in a {sentiment} {aspect}.",
        "Their {aspect} stands out as {sentiment}."
    ],
    "price": [
        "The {aspect} was {sentiment}.",
        "I found the {aspect} to be {sentiment}.",
        "Their {aspect} is {sentiment}.",
        "The {aspect} was {sentiment} for what you get.",
        "We paid a {sentiment} {aspect}.",
        "The {aspect} point is {sentiment}.",
        "Their {aspect} range is {sentiment}.",
        "For the quality, the {aspect} was {sentiment}.",
        "The {aspect} to quality ratio is {sentiment}.",
        "I think the {aspect} was {sentiment}."
    ],
    "location": [
        "The {aspect} was {sentiment}.",
        "I found the {aspect} to be {sentiment}.",
        "Their {aspect} is {sentiment}.",
        "The restaurant's {aspect} is {sentiment}.",
        "We liked the {sentiment} {aspect}.",
        "The {aspect} makes it {sentiment} to visit.",
        "Their {aspect} is {sentiment} for visitors.",
        "The {aspect} of this place is {sentiment}.",
        "Being in this {aspect} is {sentiment}.",
        "They have a {sentiment} {aspect}."
    ]
}

# Templates for implicit aspect mentions
IMPLICIT_TEMPLATES = {
    "food": [
        "I couldn't stop eating until my plate was empty. {sentiment}!",
        "Every bite was {sentiment}.",
        "I'm still thinking about what I ate there. {sentiment}!",
        "My taste buds were {sentiment_adj}.",
        "We licked our plates clean. {sentiment}!",
        "I would drive across town just to eat there again. {sentiment_adj}!",
        "The chef clearly knows what they're doing. {sentiment}!",
        "I took home leftovers for tomorrow. {sentiment}!",
        "The flavors were {sentiment_adj}.",
        "I'd recommend trying anything on their menu. {sentiment}!"
    ],
    "service": [
        "We never had to wait for anything. {sentiment}!",
        "They made us feel like royalty. {sentiment}!",
        "We were acknowledged immediately upon entering. {sentiment}!",
        "They anticipated our needs before we asked. {sentiment}!",
        "We were treated like family. {sentiment}!",
        "I had to wave my arms to get attention. {sentiment}!",
        "We waited 20 minutes before anyone took our order. {sentiment}!",
        "They rushed us through our meal. {sentiment}!",
        "The staff seemed to genuinely care about our experience. {sentiment}!",
        "They went above and beyond for our special occasion. {sentiment}!"
    ],
    "ambience": [
        "I felt relaxed the moment I walked in. {sentiment}!",
        "The noise level made conversation difficult. {sentiment}!",
        "We could hear ourselves think. {sentiment}!",
        "The place had a wonderful energy. {sentiment}!",
        "It felt like being in someone's living room. {sentiment}!",
        "The restaurant had character and charm. {sentiment}!",
        "We didn't want to leave. {sentiment}!",
        "The place felt cold and uninviting. {sentiment}!",
        "The environment put us in a good mood. {sentiment}!",
        "It was the perfect setting for our evening. {sentiment}!"
    ],
    "price": [
        "We got a lot for what we paid. {sentiment}!",
        "My wallet didn't feel much lighter after the meal. {sentiment}!",
        "I had to check my credit card statement twice. {sentiment}!",
        "You get what you pay for here. {sentiment}!",
        "I'll need to save up before coming back. {sentiment}!",
        "I didn't feel guilty about the bill. {sentiment}!",
        "It won't break the bank to eat here. {sentiment}!",
        "The experience was worth every penny. {sentiment}!",
        "I expected to pay more for the quality. {sentiment}!",
        "I left a big tip because the value was so good. {sentiment}!"
    ],
    "location": [
        "We had no trouble finding parking. {sentiment}!",
        "It's a bit out of the way, but worth the trip. {sentiment}!",
        "I wish it was closer to my home. {sentiment}!",
        "We walked there from our hotel. {sentiment}!",
        "It's hidden in a spot you'd never expect. {sentiment}!",
        "I drove past it three times before finding it. {sentiment}!",
        "It's right in the heart of everything. {sentiment}!",
        "You need to know where you're going to find this place. {sentiment}!",
        "It's so close to everything else we wanted to visit. {sentiment}!",
        "We could enjoy the neighborhood before and after our meal. {sentiment}!"
    ]
}

def generate_sentiment_value(polarity):
    """Generate a sentiment expression based on polarity."""
    if polarity == "positive":
        return random.choice(POSITIVE_EXPRESSIONS)
    elif polarity == "negative":
        return random.choice(NEGATIVE_EXPRESSIONS)
    else:
        return random.choice(NEUTRAL_EXPRESSIONS)

def generate_sentiment_adjective(polarity):
    """Generate a sentiment adjective for implicit expressions."""
    if polarity == "positive":
        return "amazing" if random.random() < 0.5 else "incredible"
    elif polarity == "negative":
        return "disappointing" if random.random() < 0.5 else "frustrating"
    else:
        return "fine" if random.random() < 0.5 else "acceptable"

def generate_explicit_review(review_id):
    """Generate a review with explicit aspect mentions."""
    # Decide how many sentences (1-5)
    num_sentences = random.randint(1, 5)
    sentences = []
    aspects_info = []
    categories = []
    
    for _ in range(num_sentences):
        # Select a category and aspect
        category = random.choice(list(ASPECTS.keys()))
        aspect = random.choice(ASPECTS[category])
        
        # Select polarity
        polarity_weights = [0.6, 0.3, 0.1]  # positive, negative, neutral
        polarity = random.choices(["positive", "negative", "neutral"], weights=polarity_weights)[0]
        
        # Generate sentiment expression
        sentiment = generate_sentiment_value(polarity)
        
        # Generate sentence from template
        template = random.choice(EXPLICIT_TEMPLATES[category])
        sentence = template.format(aspect=aspect, sentiment=sentiment)
        
        # Record aspect info
        from_index = sentence.find(aspect)
        to_index = from_index + len(aspect)
        
        sentences.append(sentence)
        aspects_info.append({
            "term": aspect,
            "polarity": polarity,
            "from": str(from_index),
            "to": str(to_index)
        })
        categories.append({
            "category": category,
            "polarity": polarity
        })
    
    return {
        "id": str(review_id),
        "text": " ".join(sentences),
        "aspects_info": aspects_info,
        "categories": categories
    }

def generate_implicit_review(review_id):
    """Generate a review with implicit aspect mentions."""
    # Decide how many sentences (1-5)
    num_sentences = random.randint(1, 5)
    sentences = []
    categories = []
    
    for _ in range(num_sentences):
        # Select a category
        category = random.choice(list(ASPECTS.keys()))
        
        # Select polarity
        polarity_weights = [0.6, 0.3, 0.1]  # positive, negative, neutral
        polarity = random.choices(["positive", "negative", "neutral"], weights=polarity_weights)[0]
        
        # Generate sentiment expression
        sentiment = generate_sentiment_value(polarity)
        sentiment_adj = generate_sentiment_adjective(polarity)
        
        # Generate sentence from template
        template = random.choice(IMPLICIT_TEMPLATES[category])
        sentence = template.format(sentiment=sentiment, sentiment_adj=sentiment_adj)
        
        sentences.append(sentence)
        categories.append({
            "category": category,
            "polarity": polarity
        })
    
    return {
        "id": str(review_id),
        "text": " ".join(sentences),
        "categories": categories
    }

def create_xml_document(reviews, is_implicit=False, filename="reviews.xml"):
    """Create an XML document in SemEval format from generated reviews."""
    doc = md.getDOMImplementation().createDocument(None, "sentences", None)
    root = doc.documentElement
    
    for review in reviews:
        sentence = doc.createElement("sentence")
        sentence.setAttribute("id", review["id"])
        
        text = doc.createElement("text")
        text_content = doc.createTextNode(review["text"])
        text.appendChild(text_content)
        sentence.appendChild(text)
        
        # For explicit reviews, add aspect terms
        if not is_implicit and "aspects_info" in review:
            aspect_terms = doc.createElement("aspectTerms")
            for aspect_info in review["aspects_info"]:
                aspect_term = doc.createElement("aspectTerm")
                aspect_term.setAttribute("term", aspect_info["term"])
                aspect_term.setAttribute("polarity", aspect_info["polarity"])
                aspect_term.setAttribute("from", aspect_info["from"])
                aspect_term.setAttribute("to", aspect_info["to"])
                aspect_terms.appendChild(aspect_term)
            sentence.appendChild(aspect_terms)
        
        # Add aspect categories for both types
        aspect_categories = doc.createElement("aspectCategories")
        for category in review["categories"]:
            aspect_category = doc.createElement("aspectCategory")
            aspect_category.setAttribute("category", category["category"])
            aspect_category.setAttribute("polarity", category["polarity"])
            aspect_categories.appendChild(aspect_category)
        sentence.appendChild(aspect_categories)
        
        root.appendChild(sentence)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(doc.toprettyxml(indent="    "))

def generate_dataset(size, is_implicit, output_path):
    """Generate a dataset of given size and type."""
    reviews = []
    generator = generate_implicit_review if is_implicit else generate_explicit_review
    
    for i in tqdm(range(size), desc=f"Generating {'implicit' if is_implicit else 'explicit'} dataset of size {size}"):
        reviews.append(generator(i + 1))
    
    dataset_type = "implicit" if is_implicit else "explicit"
    filename = f"{output_path}/{dataset_type}_{size}.xml"
    create_xml_document(reviews, is_implicit, filename)
    print(f"Generated {filename}")
    return filename

def main():
    # Create directories if they don't exist
    os.makedirs("datasets/explicit_2000", exist_ok=True)
    os.makedirs("datasets/implicit_2000", exist_ok=True)
    os.makedirs("datasets/explicit_8000", exist_ok=True)
    os.makedirs("datasets/implicit_8000", exist_ok=True)
    
    # Generate datasets
    generate_dataset(2000, False, "datasets/explicit_2000")
    generate_dataset(2000, True, "datasets/implicit_2000")
    generate_dataset(8000, False, "datasets/explicit_8000")
    generate_dataset(8000, True, "datasets/implicit_8000")

if __name__ == "__main__":
    main() 