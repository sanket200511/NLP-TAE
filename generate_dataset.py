# generate_dataset.py
import pandas as pd
import numpy as np

# Curated high-diversity dataset for Indian social media comment moderation
# Across Hindi, Hinglish, Telugu, Tamil, Malayalam, Kannada, Indian English

data = [
    # --- CLEAN HINDI / HINGLISH / INDIAN COMMENTS ---
    ("नमस्ते सर, आपका यह वीडियो बहुत ज्ञानवर्धक और उपयोगी था। धन्यवाद!", "Hindi", 0, 0, 0, 0, 0, 0),
    ("क्या आप अगले वीडियो में मशीन लर्निंग के प्रोजेक्ट्स समझा सकते हैं?", "Hindi", 0, 0, 0, 0, 0, 0),
    ("बहुत ही सुंदर प्रस्तुति, मुझे आपकी व्याख्या बहुत पसंद आई।", "Hindi", 0, 0, 0, 0, 0, 0),
    ("Bhai aapka explanation bohot badhiya tha, sab samajh aa gaya!", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Sir please share the source code and dataset link in description.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Very helpful video bhai, keep it up and upload more content!", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Super anna! Video chala bagundi, inka regular ga videos cheyandi.", "Telugu", 0, 0, 0, 0, 0, 0),
    ("Romba nalla irukku bro, romba useful-a irundhadhu nandri.", "Tamil", 0, 0, 0, 0, 0, 0),
    ("Nalla video aayirunnu, valare nanni ithu share cheythathinu.", "Malayalam", 0, 0, 0, 0, 0, 0),
    ("Thumba chennagide sir, nimma explanation thumba useful aagide.", "Kannada", 0, 0, 0, 0, 0, 0),
    ("Great initiative by the government for infrastructure development.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Mujhe lagta hai ki hume is vishay par aur charcha karni chahiye.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Both teams played well today, cricket is truly unpredictable.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Bahut accha laga dekh kar, aage bhi aise videos banate rahiye.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("All religions teach peace and brotherhood in our great country.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Kya koi bata sakta hai ki exam ki dates kab aayengi?", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Hard work always pays off, congratulations to the entire team!", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Aapki awaaz bohot soothing hai, teaching style is amazing.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("I respectfully disagree with your conclusion, here are my thoughts.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Sabhi dosto ko diwali aur holi ki hardik shubhkamnayein!", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Please keep supporting local creators and Indian education channels.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Ee topic gurinchi detailed explanation ivvandi sir please.", "Telugu", 0, 0, 0, 0, 0, 0),
    ("Intha padathoda review super, unga style enikku romba pidichirukku.", "Tamil", 0, 0, 0, 0, 0, 0),
    ("Kollam bro, nalla content aanu, continue cheyyuka.", "Malayalam", 0, 0, 0, 0, 0, 0),
    ("Ee subject kooda cover maadi, student galige thumba help aagutthe.", "Kannada", 0, 0, 0, 0, 0, 0),
    ("Best explanation of NLP algorithms in simple language.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Aapne mera doubt clear kar diya, thank you so much guru ji.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Bharat Mata Ki Jai, proud of our ISRO scientists and defense forces.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Good analysis without any bias, truly appreciate honest journalism.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Ye solution mere code me successfully run ho gaya, thanks!", "Hinglish", 0, 0, 0, 0, 0, 0),

    # --- TOXIC & INSULT (MILD TO MODERATE) ---
    ("Tu bilkul pagal hai kya? Kuch bhi faltu bol raha hai.", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("चुप कर गधे, तुझे कुछ नहीं पता, बकवास बंद कर अपनी।", "Hindi", 1, 0, 0, 0, 1, 0),
    ("You are an absolute idiot who doesn't understand basic logic.", "Indian_English", 1, 0, 0, 0, 1, 0),
    ("Kitna bewakoof insan hai tu, dimag bech ke aaya hai kya?", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("Worst video ever, you have zero knowledge and just barking.", "Indian_English", 1, 0, 0, 0, 1, 0),
    ("Nuvvu oka waste fellow vi, neeku em thelidhu mooskoni kurcho.", "Telugu", 1, 0, 0, 0, 1, 0),
    ("Loose madhiri pesadha, unakku onnum theriyaadhu muttal.", "Tamil", 1, 0, 0, 0, 1, 0),
    ("Ninakku oru bodhavum illa, mandatharam parayunnath nirthu.", "Malayalam", 1, 0, 0, 0, 1, 0),
    ("Ninage thale kettideya, yenu gothilla sumne bidthiya.", "Kannada", 1, 0, 0, 0, 1, 0),
    ("Tere jaise chomu ko kisne YouTube pe aane diya be?", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("Total clown behavior, go back to school and learn something.", "Indian_English", 1, 0, 0, 0, 1, 0),
    ("Apna ghatiya gyaan apne paas rakh, bilkul loser aadmi hai.", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("You are dumb, shut your mouth and delete your account.", "Indian_English", 1, 0, 0, 0, 1, 0),
    ("Bhai tu kitna bada feku hai, jhooth bolna band kar.", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("Muttal mathiri video podadha, unakku arive illa.", "Tamil", 1, 0, 0, 0, 1, 0),
    ("Chi chi neelanti vedhava ni eppudu chudaledhu ra.", "Telugu", 1, 0, 0, 0, 1, 0),

    # --- OBSCENE / PROFANITY / ABUSIVE ---
    ("Chup bsdk, apni aukat me reh warna teri maa chod dunga.", "Hinglish", 1, 1, 1, 1, 1, 0),
    ("कुत्ते के पिल्ले, हरामी साले तेरी गांड में गोली मार दूंगा।", "Hindi", 1, 1, 1, 1, 1, 0),
    ("Madarchod saale, aisi gaand faadunga ki rone layak nahi bachega.", "Hinglish", 1, 1, 1, 1, 1, 0),
    ("Teri behen ki chhut, saale bhadwe tujhe to raste me pelenge.", "Hinglish", 1, 1, 1, 1, 1, 0),
    ("Lodu bhosdike, apni bakwas gaand me daal le harami.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Kutte kamine, tere jaise suar ko zinda jala dena chahiye.", "Hinglish", 1, 1, 0, 1, 1, 0),
    ("Dengey ra lanjakodaka, ne amma dengi pampistha kukka.", "Telugu", 1, 1, 1, 1, 1, 0),
    ("Thevidiya paiya, oombu da sunni un vaayila soothu adippen.", "Tamil", 1, 1, 1, 1, 1, 0),
    ("Myre thayoli, ninte thala vettum njan kandaraoli.", "Malayalam", 1, 1, 1, 1, 1, 0),
    ("Sulemagane, ninna thunne cut maadi nin hathira thintisthini.", "Kannada", 1, 1, 1, 1, 1, 0),
    ("Shut the fuck up you motherfucker bastard asshole.", "Indian_English", 1, 1, 1, 0, 1, 0),
    ("Saale raand ke bacche, tera pura khandan chor hai bhenchod.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Gandu saale, tere mu me moot dunga bhikhari.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Nee lanjoduku vi ra, ninnu nariki champestha chudu.", "Telugu", 1, 1, 1, 1, 1, 0),
    ("Poda punda mavaney, unnai vettama vida maaten.", "Tamil", 1, 1, 1, 1, 1, 0),

    # --- THREAT & VIOLENCE ---
    ("I will track your IP address, come to your home and kill you.", "Indian_English", 1, 1, 0, 1, 1, 0),
    ("Tere ghar aake tujhe jaan se maar dalunga kutte, police bhi nahi bacha payegi.", "Hinglish", 1, 1, 0, 1, 1, 0),
    ("तुझे और तेरे पूरे परिवार को खत्म कर देंगे, बच के रहना।", "Hindi", 1, 1, 0, 1, 1, 0),
    ("If I see you in public, I will break your bones and smash your skull.", "Indian_English", 1, 1, 0, 1, 1, 0),
    ("Ninnu champi road meedha padesthamu jagratha.", "Telugu", 1, 1, 0, 1, 1, 0),
    ("Unnai konnu un veetlaye potruven paathukko.", "Tamil", 1, 1, 0, 1, 1, 0),
    ("Ninne njan kollum da, ninte veettil kayari vettum.", "Malayalam", 1, 1, 0, 1, 1, 0),
    ("Ninna jeeva thegithini, nanna yeduru barbeda.", "Kannada", 1, 1, 0, 1, 1, 0),
    ("We will bomb your office and shoot every single person inside.", "Indian_English", 1, 1, 0, 1, 0, 0),
    ("Tera encounter hoga saale goli maar ke bheja uda denge.", "Hinglish", 1, 1, 0, 1, 1, 0),

    # --- IDENTITY HATE / COMMUNAL TOXICITY ---
    ("These Muslims are all terrorists and should be kicked out of India.", "Indian_English", 1, 1, 0, 0, 1, 1),
    ("ये मुल्ले देशद्रोही हैं, इनको पाकिस्तान भेजो या मार गिराओ।", "Hindi", 1, 1, 0, 1, 1, 1),
    ("All Hindus are cow piss drinkers, destroy their temples and religion.", "Indian_English", 1, 1, 1, 1, 1, 1),
    ("इन अंधभक्तों और गोबर भक्तों को देश से निकालो ये कीड़े हैं।", "Hindi", 1, 0, 0, 0, 1, 1),
    ("Sikhs are all Khalistani terrorists, crush them like 1984.", "Indian_English", 1, 1, 0, 1, 1, 1),
    ("Dalits and lower caste people are dirty and don't deserve reservations.", "Indian_English", 1, 0, 0, 0, 1, 1),
    ("Chamar aur neech jaati ke log barabar nahi baith sakte.", "Hinglish", 1, 0, 0, 0, 1, 1),
    ("South Indians are all black monkeys who don't know Hindi.", "Indian_English", 1, 0, 0, 0, 1, 1),
    ("North Indian gutka eaters coming to destroy our state and culture.", "Indian_English", 1, 0, 0, 0, 1, 1),
    ("Northeast people are Chinese momo sellers, not real Indians.", "Indian_English", 1, 0, 0, 0, 1, 1),
    ("Eelanti matham vallani motham lepeyyali ee desham nunchi.", "Telugu", 1, 1, 0, 1, 1, 1),
    ("Intha mathatha azhikanum, ivanunga naatukke aabathu.", "Tamil", 1, 1, 0, 1, 1, 1),
    ("Ee jathikare ellam ivide ninnu odikkanam.", "Malayalam", 1, 0, 0, 0, 1, 1),
    ("Ee mathadavara mele attack maadi bayapadisabeku.", "Kannada", 1, 1, 0, 1, 1, 1),
    ("Terrorist jihadi kutte, tum sabko line me khada karke goli marenge.", "Hinglish", 1, 1, 0, 1, 1, 1),
    ("Katuwe saale, tumhara dharam hi nafrat sikhata hai.", "Hinglish", 1, 0, 1, 0, 1, 1),
    ("Sanghi terror supporters deserve no human rights.", "Indian_English", 1, 0, 0, 0, 1, 1),

    # --- CODE-MIXED / HINGLISH SLUR VARIATIONS ---
    ("Bhai tu itna bada chutiya hai ki tere baap ko bhi sharam aati hogi.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Saala kutta bhadwa randi rona band kar apna.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Abe o gadhapan mat kar, do thappad marunga sab nikal jayega.", "Hinglish", 1, 0, 0, 1, 1, 0),
    ("Chal nikal lodu, apni aukaat me reh kar baat kar.", "Hinglish", 1, 0, 1, 0, 1, 0),
    ("Tera channel hack karke delete kar dunga suar ke bacche.", "Hinglish", 1, 1, 0, 1, 1, 0),
    ("Dimaag ka dahi mat kar chutiye, bakwaas video banata hai.", "Hinglish", 1, 0, 1, 0, 1, 0),
    ("Shakal dekh apni kutte jaisi, aagya gyaan pelne yahan pe.", "Hinglish", 1, 0, 0, 0, 1, 0),
    ("Bsdk sale teri aukaat kya hai? do kaudi ka insaan hai tu.", "Hinglish", 1, 1, 1, 0, 1, 0),
    ("Bhadwe jaake pehle bolna seekh, chapri youtuber.", "Hinglish", 1, 0, 1, 0, 1, 0),
    ("Randi baaz saale, tujhe to jooto se peetna chahiye.", "Hinglish", 1, 1, 1, 1, 1, 0),

    # --- MORE DIVERSE CLEAN SAMPLES ---
    ("Please upload next part soon, waiting eagerly for the tutorial.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Aapki teaching methodology sach me lajawab hai, dhanyawad sir.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("India is progressing so well in science and technology.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Chala baga explain chesaru bro, doubts anni clear ayyayi.", "Telugu", 0, 0, 0, 0, 0, 0),
    ("Romba nandri sir, unga video paathu project complete pannitten.", "Tamil", 0, 0, 0, 0, 0, 0),
    ("Valare nalla aashayam aanu ithu, elavarkkum upakarapedum.", "Malayalam", 0, 0, 0, 0, 0, 0),
    ("Nimma channel naanu subscribe madidini, thumba information sigutthe.", "Kannada", 0, 0, 0, 0, 0, 0),
    ("Let's maintain dignity and respect while commenting on public forums.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Bhai sound quality improve karo thoda, baaki content top notch hai.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Thank you for raising awareness on mental health in India.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Bharat ke sabhi nagrikon ko aapas me prem se rehna chahiye.", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("This code worked like a charm on my Windows setup, cheers!", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Kya aap FastText aur BERT ke differences pe ek video bana sakte ho?", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Respect from Tamil Nadu to all Indian developers!", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Love and regards from Kerala, amazing work brother.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Hydearabad youth supports your AI educational series.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Karnataka students really benefit from these NLP tutorials.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Delhi me AI hackathon me ye logic bohot kaam aaya, shukriya!", "Hinglish", 0, 0, 0, 0, 0, 0),
    ("Clean and polite discussion always leads to better solutions.", "Indian_English", 0, 0, 0, 0, 0, 0),
    ("Hum sab milkar ek behtar aur surakshit internet bana sakte hain.", "Hinglish", 0, 0, 0, 0, 0, 0)
]

# Multiply and create rich synthetic variations with common phonetic spellings and prefixes
rows = []
for item in data:
    text, lang, t, st, obs, thr, ins, idh = item
    rows.append({
        "comment_text": text,
        "language": lang,
        "toxic": t,
        "severe_toxic": st,
        "obscene": obs,
        "threat": thr,
        "insult": ins,
        "identity_hate": idh
    })

# Add slight conversational variations
prefixes = ["", "Bro ", "Sir, ", "Abe sun, ", "Arre yaar, ", "Listen, "]
suffixes = ["", " please check.", "!!", " samajh gaya?", " bhai.", "...", "😡", "🙏", "💯"]

for item in data:
    text, lang, t, st, obs, thr, ins, idh = item
    if t == 1:
        v1 = f"{text}!!"
        v2 = f"Abe {text.lower()}"
        rows.append({"comment_text": v1, "language": lang, "toxic": t, "severe_toxic": st, "obscene": obs, "threat": thr, "insult": ins, "identity_hate": idh})
        rows.append({"comment_text": v2, "language": lang, "toxic": t, "severe_toxic": st, "obscene": obs, "threat": thr, "insult": ins, "identity_hate": idh})
    else:
        v1 = f"Bro {text}"
        v2 = f"{text} Thank you!"
        rows.append({"comment_text": v1, "language": lang, "toxic": t, "severe_toxic": st, "obscene": obs, "threat": thr, "insult": ins, "identity_hate": idh})
        rows.append({"comment_text": v2, "language": lang, "toxic": t, "severe_toxic": st, "obscene": obs, "threat": thr, "insult": ins, "identity_hate": idh})

df = pd.DataFrame(rows)
df.drop_duplicates(subset=["comment_text"], inplace=True)
df.to_csv(r"d:\Projects\Toxic-Comment-Detector\data\indian_toxic_comments.csv", index=False, encoding="utf-8")
print(f"Saved {len(df)} Indian comments to data/indian_toxic_comments.csv")
