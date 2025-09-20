#!/usr/bin/env python3
"""
Ultra-Tight Role-Based Metaphor Extractor for Echo Dot

Only extracts genuine role-based metaphors using validated patterns.
Addresses ChatGPT's feedback about loose extraction.
"""

import pandas as pd
import re
import csv
from collections import Counter, defaultdict

# Optional NLP imports - graceful fallback
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    NLP_AVAILABLE = True
    print("NLP libraries loaded")
except ImportError as e:
    NLP_AVAILABLE = False
    print(f"NLP libraries not available: {e}")

class UltraTightMetaphorExtractor:
    """Ultra-tight extractor focused only on genuine role metaphors"""
    
    def __init__(self):
        self.results = []
        self.rejected = []
        
        # Known role words for validation (not for extraction bias)
        self.known_roles = {
            # Social companions
            'friend', 'buddy', 'companion', 'partner', 'sidekick', 'pal',
            
            # Care/service roles  
            'assistant', 'helper', 'butler', 'maid', 'housekeeper', 'servant', 'concierge',
            'nanny', 'babysitter', 'caretaker', 'guardian', 'sitter',
            
            # Teaching/guidance
            'teacher', 'tutor', 'coach', 'mentor', 'guide', 'advisor', 'counselor', 'therapist',
            
            # Family roles
            'child', 'baby', 'sibling', 'roommate', 'housemate', 'family member',
            
            # Entertainment
            'dj', 'entertainer', 'storyteller', 'host', 'mc', 'radio host',
            
            # Pets/animals
            'pet', 'dog', 'cat', 'parrot', 'animal', 'creature',
            
            # Abstract/magical
            'genie', 'wizard', 'oracle', 'mind reader', 'magic', 'fairy'
        }
        
        # Initialize NLP if available
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("NLP models loaded")
            except Exception as e:
                print(f"NLP models failed: {e}")
                self.nlp = None
                self.embedder = None
        else:
            self.nlp = None
            self.embedder = None
    
    def extract_from_text(self, text, row_id):
        """Extract only validated role-based metaphors"""
        if not text or len(text.strip()) < 15:
            return
            
        original_text = text
        text = text.lower().strip()
        
        # Comprehensive patterns for role metaphor extraction
        patterns = [
            # "like having a [ROLE]" - most reliable pattern
            (r'\blike\s+having\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s+in\s+[a-z]+)?(?:\s|[.!?]|$)', 'like_having'),
            
            # "it's like having a [ROLE]"
            (r'\b(it\'?s|it\s+is)\s+like\s+having\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s+in\s+[a-z]+)?(?:\s|[.!?]|$)', 'like_having_extended'),
            
            # "it's like a/my [ROLE]" - explicit comparison
            (r'\b(it\'?s|it\s+is)\s+like\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'like_comparison'),
            
            # "it feels like a/my [ROLE]" + "seems like" with intensifiers
            (r'\b(it|this|that)\s+(feels?|seems?)\s+(?:really|basically|kind\s+of|sort\s+of|almost|literally|pretty\s+much)?\s*like\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'feels_seems_like'),
            
            # Plain copula with intensifiers - expanded
            (r'\b(it|this|that|alexa|echo\s+dot)\s+(is|was|feels?|seems?)\s+(?:really|basically|kind\s+of|sort\s+of|almost|literally|pretty\s+much)?\s*(a|an|the|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'plain_copula'),
            
            # Gendered pronouns for Alexa
            (r'\b(she\'?s|she\s+is)\s+(?:really|basically|kind\s+of|sort\s+of|almost|literally|pretty\s+much)?\s*(my|our|a|an)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'gendered_pronoun'),
            
            # Appositive patterns - "Alexa, our butler, ..."
            (r'\b(alexa|echo\s+dot|it|this|that)\s*,\s*(a|an|the|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})\s*,', 'appositive'),
            
            # "Become/turn into" role transformations
            (r'\b(has|have)?\s*become\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'become'),
            (r'\bturn(?:s|ed)?\s+into\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})(?:\s|[.!?]|$)', 'turn_into'),
            
            # Comparative "more like X than Y" patterns
            (r'\bmore\s+like\s+(a|an|my|our)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})\s+than(?:\s|[.!?])', 'contrast'),
            
            # Possessive copula for known roles
            (r'\b(it|this|that)\s+(is|was)\s+(my|our)\s+(friend|buddy|companion|assistant|helper|butler|nanny|pet|genie|guardian|mentor|coach|tutor|guide|therapist|counselor)(?:\s|[.!?]|$)', 'copula_possessive')
        ]
        
        for pattern_regex, pattern_name in patterns:
            matches = re.finditer(pattern_regex, text, re.IGNORECASE)
            
            for match in matches:
                # Extract the role phrase based on pattern type
                if pattern_name == 'copula_possessive':
                    phrase = match.group(4).strip()  # The specific role word
                elif pattern_name in ['plain_copula', 'feels_seems_like']:
                    phrase = match.group(4).strip()  # Fourth group (after intensifiers)
                elif pattern_name == 'gendered_pronoun':
                    phrase = match.group(3).strip()  # Third group for gendered
                elif pattern_name == 'appositive':
                    phrase = match.group(3).strip()  # Third group for appositive
                elif pattern_name in ['become', 'turn_into', 'contrast']:
                    phrase = match.groups()[-1].strip()  # Last capture group
                else:
                    phrase = match.groups()[-1].strip()  # Default: last capture group
                
                full_match = match.group(0)
                
                # Clean up phrases that end with location words or fragments
                phrase = re.sub(r'\s+in\s+[a-z]+$', '', phrase).strip()
                phrase = re.sub(r'\s+(of|to|when|after).*$', '', phrase).strip()
                phrase = re.sub(r'\s+has\s+learned.*$', '', phrase).strip()
                phrase = re.sub(r'\s+who.*$', '', phrase).strip()
                
                # Apply ultra-tight validation
                if self.is_genuine_role_metaphor(phrase, full_match, original_text):
                    self.results.append({
                        'phrase': phrase,
                        'quote': full_match,
                        'context': original_text[:300] + "..." if len(original_text) > 300 else original_text,
                        'pattern': pattern_name,
                        'row_id': row_id
                    })
    
    def is_genuine_role_metaphor(self, phrase, quote, full_text):
        """Ultra-tight validation for genuine role metaphors"""
        
        # Must be 1-3 words max for roles
        words = phrase.split()
        if len(words) == 0 or len(words) > 3:
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'wrong_length'})
            return False
        
        # Ban ordinal numbers
        ordinals = {'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'}
        if phrase.lower() in ordinals:
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'ordinal_number'})
            return False
        
        # Ban standalone adjectives, but allow adjective+noun combos where head is role-like
        standalone_adjectives = {
            'best', 'good', 'great', 'excellent', 'perfect', 'amazing', 'awesome', 'wonderful',
            'bad', 'terrible', 'horrible', 'awful', 'worst', 'disappointing',
            'little', 'big', 'huge', 'small', 'tiny', 'large', 'massive',
            'new', 'old', 'latest', 'newest', 'older', 'recent',
            'only', 'main', 'primary', 'secondary', 'additional', 'extra',
            'nice', 'pretty', 'beautiful', 'ugly', 'decent', 'fine', 'okay',
            'cheap', 'expensive', 'affordable', 'costly', 'pricey',
            'easy', 'hard', 'difficult', 'simple', 'complex', 'complicated',
            'quick', 'fast', 'slow', 'rapid', 'speedy',
            'smart', 'intelligent', 'stupid', 'dumb', 'clever',
            'useful', 'useless', 'helpful', 'handy', 'convenient'
        }
        
        # Only block standalone adjectives, not adjective+noun combos
        if phrase.lower() in standalone_adjectives:
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'standalone_adjective'})
            return False
        
        # Ban abstract concepts that aren't roles
        abstract_non_roles = {
            'must', 'need', 'want', 'wish', 'dream', 'hope',
            'bit', 'lot', 'bunch', 'ton', 'pile', 'stack',
            'waste', 'loss', 'gain', 'benefit', 'advantage', 'disadvantage',
            'start', 'beginning', 'end', 'finish', 'completion',
            'try', 'attempt', 'effort', 'shot', 'go', 'chance',
            'way', 'method', 'approach', 'technique', 'style',
            'thing', 'stuff', 'item', 'object', 'piece',
            'understanding', 'knowledge', 'information', 'data', 'fact',
            'artificial intelligence person', 'ai person', 'intelligence person',
            'number', 'wonder', 'miracle', 'dream', 'nightmare',
            'better', 'worse', 'best', 'worst', 'bigger', 'smaller',
            'disappointment', 'surprise', 'shock', 'revelation',
            # Additional non-roles identified
            'minor', 'major', 'light eater', 'heavy eater', 'eater',
            'matter', 'complement', 'compliment', 'supplement', 'holder',
            'lot nicer', 'bit nicer', 'little brighter', 'little louder', 'little slower'
        }
        
        if phrase.lower() in abstract_non_roles:
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'abstract_non_role'})
            return False
        
        # Ban phrases that are clearly not roles (physical descriptions, comparatives, fragments)
        non_role_patterns = [
            r'\b(bit|little bit|tad bit)\s+(bigger|smaller|louder|quieter|brighter|darker|larger)\b',
            r'\b(big|huge|small|tiny)\s+(disappointment|surprise|mistake|problem)\b',
            r'\b(charcoal|black|white|red|blue)\s+(color|colour)\b',
            r'\b(programming|technical|software)\s+(issue|problem|error)\b',
            r'\b(weird|strange|odd)\s+(incident|event|thing)\b',
            r'\b(fifth|sixth|seventh|eighth|ninth|tenth)\s+(one|time)\b',
            # Ban comparative/descriptive fragments
            r'\b(much|way|far)\s+(better|worse|more|less)\s+(voice|sound|quality|proficient)\b',
            r'\b(fun|cute|nice|great)\s+and\s+(convenient|easy|simple)\b',
            r'\b(copy\s+cat|copycat)\s+of\b',
            r'\b(no-brainer|brainer)\b',
            r'\b(one\s+you\s+want|thing\s+we\s+never)\b',
            r'\b(response\s+when|tradeoff\s+to)\b',
            r'\b(perfect|great|excellent)\s+(economical|economic)\s+(answer|solution)\b',
            r'\b(really|very|pretty)\s+(cute|nice|good)\s+(holder|stand|mount)\b',
            r'\b(perfect|great|excellent)\s+(holder|complement|compliment|supplement)\b',
            r'\b(lot|bit|little)\s+(nicer|better|worse|brighter|louder|slower|faster)\b',
            r'\b(very|really|quite)\s+(positive|negative)\s+(customer|experience|review)\b'
        ]
        
        for pattern in non_role_patterns:
            if re.search(pattern, phrase.lower()):
                self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'non_role_pattern'})
                return False
        
        # Ban tautologies
        if any(word in phrase.lower() for word in ['echo', 'dot', 'alexa', 'amazon']):
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'tautology'})
            return False
        
        # Ban technical/product terms and non-role phrases
        technical_terms = {
            'product', 'device', 'gadget', 'machine', 'appliance', 'tool',
            'speaker', 'timer', 'alarm', 'clock', 'radio', 'player',
            'gift', 'present', 'purchase', 'buy', 'deal', 'bargain',
            'replacement', 'upgrade', 'downgrade', 'update', 'version',
            'model', 'generation', 'gen', 'series', 'type', 'kind',
            'christmas', 'birthday', 'holiday', 'occasion', 'event',
            'winner', 'loser', 'hit', 'miss', 'success', 'failure',
            'improvement', 'enhancement', 'addition', 'accessory',
            'fit', 'match', 'choice', 'option', 'alternative'
        }
        
        # Check if phrase contains technical terms
        phrase_words = phrase.lower().split()
        if any(word in technical_terms for word in phrase_words):
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'technical_term'})
            return False
        
        # For copula patterns, be extra strict - reject phrases ending with "for"
        if 'for' in quote.lower() and phrase.endswith('for'):
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'incomplete_phrase_for'})
            return False
        
        # Reject phrases that are clearly product descriptions, not roles
        product_description_patterns = [
            r'\b(great|good|nice|amazing|excellent|perfect|best|worst)\s+(little\s+)?speaker\b',
            r'\b(gift|present)\s+for\b',
            r'\b(product|device|item)\s+for\b',
            r'\b(replacement|upgrade|improvement)\s+for\b',
            r'\b(accessory|addition)\s+for\b',
            r'\b(fit|match|choice)\s+for\b',
            r'\b(voice\s+)?(recognition|control|command)\b',
            r'\b(remote|controller|handler)\b',
            r'\b(investment|purchase|buy|deal)\b'
        ]
        
        for pattern in product_description_patterns:
            if re.search(pattern, phrase.lower()):
                self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'product_description'})
                return False
        
        
        # Ban literal function references
        if any(literal in full_text.lower() for literal in ['looks like', 'sounds like', 'use as', 'set as', 'works as', 'work as']):
            self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'literal_function'})
            return False
        
        # Additional role-like phrases not in main list
        additional_roles = {
            'life saver', 'lifesaver', 'savior', 'saviour',
            'guardian angel', 'angel',
            'sidekick', 'partner in crime',
            'voice in the room', 'voice',
            'presence', 'company'
        }
        
        # Allow known roles and additional roles
        if phrase.lower() in self.known_roles or phrase.lower() in additional_roles:
            return True
        
        # Allow clear role morphology (ends in -er, -or, -ist, etc.)
        if self.has_role_morphology(phrase):
            return True
        
        # Allow compound phrases if they contain a known role word
        phrase_words = phrase.lower().split()
        if len(phrase_words) <= 3:
            for word in phrase_words:
                if word in self.known_roles or word in additional_roles:
                    return True
        
        # IMPROVED: Allow single-word roles if they have determiner evidence in quote
        if len(phrase_words) == 1:
            # Check if quote shows determiner evidence: "a/an/the/my/our [role]"
            if any(det in quote.lower() for det in [' a ', ' an ', ' the ', ' my ', ' our ']):
                # Additional single-word roles that are clearly role-like (including hyphenated)
                single_word_roles = {
                    'companion', 'guardian', 'mentor', 'advisor', 'counselor', 'therapist',
                    'butler', 'maid', 'servant', 'concierge', 'doorman', 'bouncer',
                    'teacher', 'tutor', 'coach', 'trainer', 'instructor', 'professor',
                    'guide', 'navigator', 'pilot', 'driver', 'chauffeur',
                    'entertainer', 'performer', 'comedian', 'storyteller', 'narrator',
                    'secretary', 'receptionist', 'operator', 'dispatcher',
                    'babysitter', 'caretaker', 'caregiver', 'nurse', 'doctor',
                    'translator', 'interpreter', 'mediator', 'negotiator',
                    'watchman', 'sentry', 'guard', 'protector', 'defender',
                    # Hyphenated compound roles
                    'life-saver', 'time-saver', 'story-teller', 'care-giver', 'baby-sitter',
                    'house-keeper', 'gate-keeper', 'peace-keeper', 'goal-keeper'
                }
                
                if phrase.lower() in single_word_roles:
                    return True
        
        # Use NLP for final validation if available
        if self.nlp and self.embedder:
            if self.nlp_validates_as_role(phrase):
                return True
        
        # Reject everything else
        self.rejected.append({'phrase': phrase, 'quote': quote, 'reason': 'not_role_like'})
        return False
    
    def has_role_morphology(self, phrase):
        """Check if phrase has role-like word endings"""
        words = phrase.split()
        if not words:
            return False
            
        # Check the head word for role-like endings
        head_word = words[-1].lower()
        
        # Common role suffixes
        role_suffixes = ['-er', '-or', '-ist', '-ian', '-ant', '-ent', '-man', '-person']
        
        for suffix in role_suffixes:
            if head_word.endswith(suffix[1:]):  # Remove the dash
                return True
        
        return False
    
    def nlp_validates_as_role(self, phrase):
        """Use NLP to validate if phrase semantically represents a role"""
        try:
            # Immediate reject for obvious non-roles
            non_role_words = {
                'bomb', 'blast', 'fun', 'pain', 'one', 'favorite', 'hit', 'winner',
                'gem', 'steal', 'deal', 'buy', 'purchase', 'product', 'device',
                'thing', 'toy', 'gadget', 'machine', 'tool', 'item', 'piece'
            }
            
            if phrase.lower().strip() in non_role_words:
                return False
            
            # Check semantic similarity to known roles
            phrase_embedding = self.embedder.encode([phrase])
            role_embeddings = self.embedder.encode(list(self.known_roles))
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(phrase_embedding, role_embeddings)[0]
            max_similarity = max(similarities)
            
            # Much more strict threshold for role-likeness
            return max_similarity > 0.6
            
        except Exception as e:
            print(f"NLP validation error: {e}")
            return False
    
    def process_csv(self, filepath, scope='hard+soft'):
        """Process CSV with ultra-tight extraction"""
        print(f"Loading: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} rows")
        except Exception as e:
            print(f"Error: {e}")
            return
        
        # Apply scope
        if scope == 'hard':
            mask = df['topic_confidence'] == 'hard'
        elif scope == 'hard+soft':
            mask = df['topic_confidence'].isin(['hard', 'soft'])
        else:
            mask = pd.Series([True] * len(df))
        
        df_filtered = df[mask]
        print(f"Scope '{scope}': {len(df_filtered)} rows")
        
        # Extract metaphors
        for idx, row in df_filtered.iterrows():
            text = str(row.get('text', ''))
            self.extract_from_text(text, idx)
            
            if idx % 2000 == 0:
                print(f"Processed {idx}...")
        
        print(f"Found {len(self.results)} candidates, rejected {len(self.rejected)}")
    
    def merge_similar_phrases(self, results):
        """Merge conceptually similar phrases for better reporting"""
        # Define conceptual groups for cleaner reporting
        groups = {
            'friend': ['friend', 'best friend', 'good friend', 'new friend', 'virtual friend', 'smart funny friend', 'best friend living', 'close friend', 'new best friend', 'best friend during', 'best friend in'],
            'assistant': ['assistant', 'personal assistant', 'own personal assistant', 'personal assistant at', 'virtual assistant', 'digital voice assistant', 'bedside assistant that', 'entertaining assistant to', 'great assistant', 'great assistant in', 'wonderful assistant'],
            'buddy': ['buddy', 'little buddy', 'best buddy', 'buddy around when', 'good buddy', 'buddy while i'],
            'companion': ['companion', 'little companion', 'good companion', 'perfect companion', 'constant companion', 'great companion', 'great companion once', 'great companion who', 'true companion', 'perfect companion to'],
            'helper': ['helper', 'little helper', 'personal helper', 'helpful assistant', 'best helper in', 'great little helper'],
            'genie': ['genie', 'second genie along'],
            'saver': ['life saver', 'time saver', 'sanity saver'],
            'partner': ['partner', 'very usefull partner', 'sidekick', 'perfect sidekick to'],
            'servant': ['butler', 'maid', 'servant', 'concierge', 'secretary', 'helpful servant who']
        }
        
        merged = {}
        for result in results:
            phrase = result['phrase']
            
            # Find which group this phrase belongs to
            group_key = phrase  # default to itself
            for group_name, variants in groups.items():
                if phrase in variants:
                    group_key = group_name
                    break
            
            if group_key not in merged:
                merged[group_key] = {
                    'phrases': [],
                    'count_total': 0,
                    'examples': []
                }
            
            merged[group_key]['phrases'].append(phrase)
            merged[group_key]['count_total'] += result['count_total']
            merged[group_key]['examples'].append(result)
        
        # Convert back to list format
        final_results = []
        for group_key, data in merged.items():
            # Use the shortest example quote
            best_example = min(data['examples'], key=lambda x: len(x['example_quote']))
            
            final_results.append({
                'phrase': group_key,
                'count_total': data['count_total'],
                'example_quote': best_example['example_quote'],
                'pattern_type': best_example['pattern_type'],
                'variants': ', '.join(sorted(set(data['phrases'])))
            })
        
        return sorted(final_results, key=lambda x: (-x['count_total'], x['phrase']))

    def generate_output(self, output_path='results.csv', rejected_path='rejected.csv', min_count=1):
        """Generate final validated results with conceptual merging"""
        
        if not self.results:
            print("No valid role metaphors found!")
            return []
        
        # Aggregate by phrase
        phrase_counts = Counter(r['phrase'] for r in self.results)
        phrase_examples = {}
        
        for result in self.results:
            phrase = result['phrase']
            if phrase not in phrase_examples or len(result['quote']) < len(phrase_examples[phrase]['quote']):
                phrase_examples[phrase] = result
        
        # Build raw results
        raw_results = []
        for phrase, count in phrase_counts.most_common():
            if count >= min_count:
                example = phrase_examples[phrase]
                raw_results.append({
                    'phrase': phrase,
                    'count_total': count,
                    'example_quote': example['quote'],
                    'pattern_type': example['pattern']
                })
        
        # Merge similar phrases for better conceptual reporting
        final_results = self.merge_similar_phrases(raw_results)
        
        # Add ranks
        for i, result in enumerate(final_results, 1):
            result['rank'] = i
        
        # Save results
        if final_results:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['rank', 'phrase', 'count_total', 'example_quote', 'pattern_type', 'variants'])
                writer.writeheader()
                writer.writerows(final_results)
            print(f"Results saved to: {output_path}")
        
        # Save rejected for debugging
        if self.rejected:
            with open(rejected_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['phrase', 'quote', 'reason'])
                writer.writeheader()
                writer.writerows(self.rejected)
            print(f"Rejected items saved to: {rejected_path}")
        
        # Summary
        print(f"\nFINAL RESULTS:")
        print(f"Valid role metaphors: {len(final_results)}")
        print(f"Total instances: {sum(r['count_total'] for r in final_results)}")
        print(f"Rejected: {len(self.rejected)}")
        
        if final_results:
            print(f"\nTOP ROLE METAPHORS:")
            for result in final_results[:10]:
                print(f"{result['rank']}. '{result['phrase']}' ({result['count_total']}x)")
                print(f"    Example: \"{result['example_quote']}\"")
        
        # Rejection breakdown
        if self.rejected:
            rejection_counts = Counter(r['reason'] for r in self.rejected)
            print(f"\nREJECTION REASONS:")
            for reason, count in rejection_counts.most_common():
                print(f"  {reason}: {count}")
        
        return final_results

def main():
    """Main execution with comprehensive pattern matching"""
    print("Echo Dot Role Metaphor Extractor")
    print("=" * 35)
    print("Comprehensive pattern matching with precision filtering")
    
    extractor = UltraTightMetaphorExtractor()
    
    # Process data
    extractor.process_csv('clean_text.csv', scope='hard+soft')
    
    # Generate validated results
    results = extractor.generate_output('results.csv', 'rejected.csv', min_count=1)
    
    print(f"\nAnalysis complete. Check results.csv for genuine role metaphors.")

if __name__ == "__main__":
    main()