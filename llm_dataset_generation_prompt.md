# LLM Prompt for Restaurant Review Dataset Generation

## Task Description

Generate a dataset of restaurant reviews with both explicit and implicit aspect mentions in SemEval-style XML format. This dataset will be used for aspect-based sentiment analysis research.

## Concepts

- **Explicit Aspects**: Direct mentions of features or aspects of a restaurant (e.g., "The *food* was delicious", "The *service* was slow")
- **Implicit Aspects**: Indirect references to aspects without explicitly naming them (e.g., "I couldn't stop eating" → implicit reference to food quality, "We waited 30 minutes for a table" → implicit reference to service)

## Dataset Format

Generate the reviews in SemEval XML format as follows:

```xml
<?xml version="1.0" ?>
<sentences>
    <sentence id="1">
        <text>[Review text goes here]</text>
        <aspectTerms>
            <!-- Only for explicit aspects -->
            <aspectTerm term="food" polarity="positive" from="10" to="14"/>
            <!-- Include from/to character positions -->
        </aspectTerms>
        <aspectCategories>
            <!-- For both explicit and implicit aspects -->
            <aspectCategory category="food" polarity="positive"/>
            <aspectCategory category="service" polarity="negative"/>
        </aspectCategories>
    </sentence>
    <!-- More sentences... -->
</sentences>
```

## Aspect Categories

Include aspects from these five categories:

1. **Food**: Mentions of food, dishes, taste, menu items, ingredients, portions
2. **Service**: Mentions of staff, waiters, customer service, attentiveness
3. **Ambience**: Mentions of atmosphere, decor, noise level, comfort, music
4. **Price**: Mentions of cost, value, affordability, discounts
5. **Location**: Mentions of accessibility, parking, neighborhood, convenience

## Sentiment Polarities

Use one of these sentiment polarities:
- **positive**: Expressing satisfaction or approval
- **negative**: Expressing dissatisfaction or criticism
- **neutral**: Factual or neither positive nor negative

## Instructions

1. Generate 2000 restaurant reviews, with approximately 60% positive, 30% negative, and 10% neutral polarities
2. For explicit aspect reviews, directly mention aspects from the categories
3. For implicit aspect reviews, imply aspects without explicit mentions
4. For each review:
   - Assign a unique ID
   - Include correct character positions (from/to) for explicit aspects
   - Assign the appropriate aspect categories
   - Tag the sentiment polarity correctly

## Examples

### Explicit Aspect Example

```xml
<sentence id="42">
    <text>The pasta was delicious but the service was slow.</text>
    <aspectTerms>
        <aspectTerm term="pasta" polarity="positive" from="4" to="9"/>
        <aspectTerm term="service" polarity="negative" from="27" to="34"/>
    </aspectTerms>
    <aspectCategories>
        <aspectCategory category="food" polarity="positive"/>
        <aspectCategory category="service" polarity="negative"/>
    </aspectCategories>
</sentence>
```

### Implicit Aspect Example

```xml
<sentence id="43">
    <text>I couldn't stop eating until my plate was empty. Amazing!</text>
    <aspectCategories>
        <aspectCategory category="food" polarity="positive"/>
    </aspectCategories>
</sentence>
```

## Format for Output

Please generate the complete XML document with 2000 reviews. Ensure each review has appropriate aspect terms (for explicit aspects) and aspect categories (for both explicit and implicit aspects). Calculate the "from" and "to" character positions accurately.

For generating a dataset of 2000 reviews, please create a balanced mixture of:
- Reviews with only explicit aspects
- Reviews with only implicit aspects
- Reviews with both explicit and implicit aspects
- Reviews covering all five aspect categories

Please maintain high diversity in the vocabulary, expressions, and sentence structures to create a rich, varied dataset. 