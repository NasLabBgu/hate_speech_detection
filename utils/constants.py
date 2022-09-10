import re
ECHO_UNIGRAM_TERMS = {'foreigners', 'really', 'music', 'wessberger', 'feminism', 'cnn', 'evil', 'reason', 'mark',
              'marxism', 'zuckerberg', 'going', 'viacom', 'giordano', 'globalist-funded', 'baby',
              'british', 'mouths', "who's", 'russia', 'echoes', 'katzsenberg', 'anti-globalism',
              '666', 'finance', 'dugun', 'duginist', 'victoria', 'jb', 'disease', 'vaccinations', 'conspiracy',
              'neocons', 'isis', 'q', 'journalist', 'wakeup', 'irishbit', 'believe', 'globalistpuppet', 'our',
              'diversity', 'wrong', 'j3wish', 'lawmakers', 'capitalist', 'blackrock', 'puppy',
              'bolsheviks', 'morgan', 'praeger', 'scientists', 'hillary', 'victory', 'afraid', 'capital',
              'hirsch', 'owners', 'name', 'christians', 'books', 'billmaher', 'protesters', 'free', 'kirkland',
              'censored', 'the', 'culture', 'tribe', 'yhwh', 'press', 'putin', 'cabal', 'kushner', 'religion',
              'government', 'ussr', 'coincidence', 'movies', 'by', 'klobuchar', 'zog', 'anti-zionist', 'juicy',
              'roman', 'julia', 'noticing', 'barnett', "america's", 'voted', 'marvel', 'us', 'peston',
              '@fckeveryword', 'safety', 'levin', 'marxist', 'spielberg', 'catholic', 'miscegenation', 'rabbi',
              'memed', 'democracy', 'goldberg', 'feinstein', 'mindgeek', '@nytimes', 'adl', 'colledge',
              'marrano', 'national', 'globalist-induced', 'health', 'jerusalem', 'injustice', 'lgbtp',
              '#zuckerberg', 'isreal', 'anti-semite', 'projection', 'goyim', 'circumcised', 'dnc', 'comedy',
              'yourself', 'fortune', 'catfishing', 'victim', 'globalistswho', 'unvarified', 'jews-scape-goats',
              'unmasked', 'mean', 'bonnier', 'trumpty', 'tesco', 'genoese', 'cia', 'shocker', 'solidarity',
              'google', 'frasier', 'globalists', 'we', 'their', 'governments', 'lie', 'theirs', 'your',
              '@billmaher', 'terminology', 'rx', 'piers', '404', 'tiddy', 'media', 'hollywood', 'folk',
              'zionist', 'fsction', 'bauhaus', 'salks', 'system', 'dugin', 'problem', 'homeboys', "god's",
              'frank', 'kommunister', 'fauci', 'mnuchin', 'levine', 'levy', 'few', 'orchestrator',
              'vereddeplorable', 'moron', 'controlled', 'word', 'money', 'rules', 'june', 'others', 'university',
              'rothschilds', 'charlemagne', 'beyond', 'friedman', 'institutions', '@marcorubio',
              'commie', '@borisjohnson--like', 'bloomberg', 'libs', 'sopel', 'protestant', 'hell',
              'them', "they're", 'conservatives', 'overlords', 'pagans', 'friends', 'haim', 'usa', 'jewish',
              'forces', '@billgates', 'antisemitism', '#climate', 'god', 'democrats',
              'judeochristian', "kramer's", 'rats', 'people', 'republic', 'treason', 'spooks', 'mel',
              'american', 'replacement', 'nobody', "you're", 'disraeli', 'tribesman', 'felix',
              'philanthropists', 'anglosphere', 'globalist-imposed', 'dots', '#unipcc', 'wealthy',
              'cedars-sinai', 'cases', 'psyops', 'jew', 'subvertor', 'americans', 'white', 'whites',
              'parthians', 'experts', 'wicca', 'lsd', 'communist-owned', 'prageru', 'house',
              '@benshapiro', 'globalism', 'live', 'chosenites', 'isn’twhite', 'preachers',
              'talmudic', 'yeehawpill', 'robinson', 'banksters', 'msm', 'lieber', 'censorship', 'kikes',
              'crypto', 'virus', 'liberal', 'twits', '#bigpharma', 'soviet', 'barbarians', 'thezionists',
              'fink', 'zionst', 'deatheaters', 'russiagate', 'thier', 'anti-globalists', 'it', '#rabiat',
              'groyper', 'yahweh', 'felsenthal', 'man', 'power', 'ugly', 'historians', 'supremacists',
              'mossad', 'neoliberalism', 'ideology', 'nwo', 'jesus', 'casualty', '#karlmarx', 'lawyers',
              'yid', "#zog's", 'js', 'donors', 'gutterman', 'intellectualism', 'schlomo', 'scrubs',
              'satanists', 'ch0000senites', 'ass', 'control', 'vaccine', 'presumed', 'crusaders', 'outsider',
              'israel', 'a-ward', 'anarchist', 'russia', 'goblins', 'shumer', 'abramowitz',
              '#gangs', 'leibovitz', 'space', 'grifting', 'racism', 'terrorists', 'those',
              'bridgitte', 'globalistsnetwork', 'daiv', 'cohencidence', 'feminists', 'russian',
              'jewishlol', 'bernstein', 'masters', 'patriarchy', 'german', 'twisted',
              'zionism', 'ong', 'elite', 'merkel', 'maitliss', 'banks', 'j’s', 'anti-globalist',
              'community', 'academic', 'john', 'liberation', 'winners', 'typical', 'jack', 'rubin', 'corona',
              'weiss', 'sebastian', 'guys', 'klein', 'weinstein', '@conservatives', 'sports', 'khazar',
              'capitalism', 'seidelbaum', 'again', 'yangs', 'zionists-', 'don’t', 'pope', 'eu', 'she', 'expose',
              'christian', 'peta', 'nato', 'non-jews', "gottfried's", 'goy', 'beria', 'bourgeoisie', 'noun',
              'communism', 'group', 'family', 'conspirators', 'lol', 'author', 'party', 'metaphor', 'allies',
              'bigtrick', '@borisjohnson', 'names', 'who', 'phoebe', 'globalnews', 'supernational',
              'legion', '#markzuckerberg', 'this', 'foreign', 'anyone', 'christianity', 'how', 'they’d',
              'ross', 'smells', 'he', 'juden', 'multiculturalism', 'anti-zionists', '#liberationday', 'fed',
              'prick', 'into', 'globalistpuppets', '#jog', 'scholar', 'yousef', 'csa', 'youtube', 'quakers',
              'alt-right', 'david', 'parentheses', 'overlord)', 'bornstein',
              "bbc's", 'semitism', 'nose', 'infiltrated', 'they’ve', 'raffa', 'yes',
              'culturalmarxists', 'sanders', 'influential', '#zog', '-lorber', 'mole', 'q-date', 'inversion',
              'values', 'merchant', 'jewlius', 'iceberg', 'abramovich', 'neoliberal', 'weissman',
              'plan', 'freemasonry', 'schweitz', 'globalist-adjacent', 'these', 'thou', 'chomsky',
              'nah', 'coronavirus', 'h3h3', 'sect', 'left', 'kessler', 'filthy', 'zionists', 'skechers',
              'church', 'whitebrother', 'bbc', 'cardinals', 'israeli', 'hate_networks', 'oligarch', 'hmmmmmmmmmmmmmmmm',
              'dream', 'shilling', 'eurasianism', 'chassids', 'kike', 'marxists', 'elites', 'you’re', 'nick',
              'gerstein', 'trs', 'kristol', 'parcak', 'that', 'woke', 'cities', 'un', 'leftists', '@mayorfrey',
              'miriam', 'official', 'minions', 'bagels', 'state', 'democratic', 'enemies', 'freedom', 'jews',
              'theyyyyyy', 'ecb', 'theories', 'jewellers', 'soros)', 'sympathetic', 'kurds', 'truth', 'there',
              'punk', 'structuralism', 'boris', 'radicals', 'antifa', 'x', 'parenthesis', 'imagine',
              'rothschild', 'liberalism', 'know', 'boom', 'when', 'duplicity', 'globuhll-izm', 'sims',
              'his', 'targets', 'banker', 'chosen', 'they', '#zog-', 'her', 'philosophy', 'seth', 'reformation',
              'j', 'oss', 'neocon', 'ww2', 'theyr', 'cernovich', 'creators', 'you', 'judaism', 'jap',
              'timber', 'climate', 'communists', 'america', 'enemy',
              'satanists-globalists-talmudists-talmu-mongols-bilderbergs-idont-even-fucking-care-you-elite-cruel-and-stupid-fuckers',
              'bankers', 'certain', 're-set', 'baum', 'cost', 'liberals', 'masonic', 'humanitarianism',
              "kolomoisky's", 'bugmen', 'globalist', 'inequality', 'ventriloquists', 'soros', 'heeb', 'torba',
              'landlords', 'narrative', 'theyblame', '#msm', 'to(antisemitism)', 'non-governmental',
              'masters--not', 'they’re', 'him', 'redfish', 'english', 'stats', 'establishment', 'stupid',
              'communist', 'stern', 'weev', 'probably', 'grandpa', 'schmidberger'}

ANTISEMITIC_SLURS = {
    'jewing', 'jewrat', 'jewboy', 'jew-ing', 'israhells', 'globo', 'cabal', 'goyhatred', 'quarantinezionists',
    'israelapartheid', 'zios', 'zog', 'yid', 'heeb', 'globalist', 'globalists', 'globalism', 'talmudists', 'talmudist',
    'kabbalists', 'kabbalist', 'frankists', 'frankist', 'sabbateans', 'sabbatians', 'zoharists', 'zoharist',
    'cabal', 'goyim', 'goy', 'shekel', 'zion', 'zionism', 'zionist', 'zionists', 'kike', 'kikes', 'jewboy', 'zog',
    'marxist', 'marxists', 'establishment', 'jewess', '#cohencidence', 'cohencidence', 'talmudic', 'shekelstein',
    'namethejew', '#namethejew', '#thegoyimknow', 'schlomo', 'khazar', '#christkiller', 'christkiller', 'jesuskillers',
    'jesuskiller', 'jesus killers', 'jesus killer', '#jesuskillers', '#jesuskiller', '#jewishvirus', 'jewishvirus',
    'jewish virus'
}

OTHER_RACIAL_SLURS = {
    'chink', 'wetback', 'spic', 'half-caste', 'halfcaste', 'dago', 'kraut', 'zipperhead', 'lebaneezer'
    'nigger', 'gook', 'gooks', 'jiggaboo', 'negroid', 'shitskin', 'raccoon', 'racoon', 'darkie', 'mick',
    'polak', 'hun', 'porchmoneky', 'coon', 'alligatorbate', 'cracker', 'honkey', 'spade', 'coolie', 'wigger',
    'whitey', 'wog', 'wop', 'globo-homo', 'globohomo', 'shitlib', 'libtard', 'gook', 'gooks', 'fag', 'faggot',
    'bigot', 'spic', 'spick', 'spik', 'spig', 'spigotty', 'nigger', 'niggers', 'nigga', 'niggas', 'nigs',
    'negress', 'sheboon', 'negro', 'negroes', 'negroid', 'spearchucker', 'groid', 'jigaboo', 'jiggabo',
    'jigarooni', 'chink', 'chinky', 'cuckservative', 'honkey', 'honky', 'honkie', 'latino', 'beaner',
    'greaseball', 'wetback', 'brownie', 'greaser', 'sudaca', 'tacohead', 'cholo', 'tonk', 'mullato', 'mulatto',
    'mongoloid', 'arabush', 'redneck', 'hajji', '#jewishvirus', 'jewishvirus', 'jewish virus', '#chinesevirus',
    'chinesevirus', 'chinese virus', '#muslimvirus', 'muslimvirus', 'muslim virus', '#coronajihad',
    'coronajihad', 'corona jihad', '#christianvirus', 'christianvirus', 'christian virus', '#republicanflu',
    'republicanflu', 'republican flu', 'towel head', '#towelhead', 'towelhead', 'musrat', '#musrat', 'musrats',
    '#musrats', 'muzrat', '#muzrat', 'muzrats', '#muzrats', 'fuckmuslims', '#fuckmuslims', 'fuckmuslim',
    '#fuckmuslim', 'muslimpropaganda', '#muslimpropaganda', '#noislam', '#closedborders', '#closeborders',
    '#stopinvasion', '#defendeurope', '#defendeuropa', '#remigration', '#BackToAfrica', '#leftrad', '#leftrads',
    'leftard', 'leftards', '#ChinaLiedAndPeopleDied', '#BoycottChina', '#CCP_is_terrorist', 'tranny', 'bugland',
    'chankoro', 'chinazi', 'insectoid', 'bugmen', 'chingchong', 'dyke', 'muzzie', 'beached whale', 'asslifter',
    'tar baby', 'paki', 'bootlip', 'camel jockey', 'surrender monkey', 'hebe', 'nigglet', 'shit skin',
    'raghead', 'blue gum', 'hymie', 'sand nigger', 'goat fucker', 'niglet', 'lard ass', 'curry nigger',
    'moon cricket', 'fatso', 'camel fucker', 'coon', 'oven dodger', 'moolie', 'dune coon', 'porch monkey',
    'carpet kisser', 'mayate', 'wetback', 'mudslime', 'dindu nuffin', 'cave nigger', 'kike', 'muslimturd',
    'zipperhead', 'dindu-nuffin', 'tar-baby', 'cavenigger', 'carpetkisser', 'goatfucker', 'carpet-kisser',
    'porch-monkey', 'lard-ass', 'sandnigger', 'sand-nigger', 'oven-dodger', 'tarbaby', 'surrender-monkey',
    'lardass', 'towel head', 'porchmonkey', 'beached-whale', 'blue-gum', 'cameljockey', 'moon-cricket',
    'mooncricket', 'surrendermonkey', 'dindunuffin', 'camel-fucker', 'ovendodger', 'curry-nigger',
    'shitskin', 'dunecoon', 'bluegum', 'beachedwhale', 'dune-coon', 'camelfucker', 'currynigger',
    'goat-fucker', 'camel-jockey', 'shit-skin', 'cave-nigger', 'yellowniggers', 'yellownigger', 'streetshitter',
    'ricenigger', 'kungflu', '#kungflu', 'kung-flu', 'commiecrud', '#commiecrud', 'ching chong', 'lebanezer',
    'Ginzo', 'baracoon', 'Barracoon', '#AlligatorBait', 'AlligatorBait', 'Alligator Bait', 'golliwog', 'golliwogg',
    'pickaninny'
}


COVID_RELATED_SLURS = {
    'kungflu', 'boycottchina', 'chinaliespeopledie', 'jewishvirus', 'jewish virus', '#jewishvirus', '#muslimvirus',
    'muslimvirus', 'muslim virus', '#christianvirus', 'christianvirus', 'christian virus', '#republicanflu',
    'republicanflu', 'republican flu'
}

ALL_HATE_TERMS = ANTISEMITIC_SLURS
ALL_HATE_TERMS.update(OTHER_RACIAL_SLURS)
ALL_HATE_TERMS.update(COVID_RELATED_SLURS)
ALL_HATE_TERMS = [HATE_TERM.lower().strip() for HATE_TERM in ALL_HATE_TERMS]
UNIGRAM_HATE_TERMS = [HATE_TERM for HATE_TERM in ALL_HATE_TERMS if len(HATE_TERM.split(' ')) == 1]
BIGRAM_OR_HIGHER_HATE_TERMS = [HATE_TERM for HATE_TERM in ALL_HATE_TERMS if len(HATE_TERM.split(' ')) > 1]

TWITTER_USER_META_KEYS = ['id_str', 'name', 'screen_name', 'location', 'description', 'protected',
                  'followers_count', 'friends_count', 'listed_count', 'created_at', 'favourites_count',
                  'verified', 'statuses_count', 'lang']


# Stop words
STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
STOP_WORDS.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        STOP_WORDS.add(stopword.replace("'", apostrophe))



CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

# REGEXES
URL_RE = re.compile(
    r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
RT_RE = re.compile(r'RT\s')
PUNCT_RE = re.compile('[“",?;:!\\-\[\]_.%/\n]')
MENTION_RE = re.compile(r'@\w+')