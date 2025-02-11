# Model Performance Report: koala

## Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|--------|
| accuracy | 0.410 |
| macro_precision | 0.274 |
| macro_recall | 0.296 |
| macro_f1 | 0.243 |
| avg_latency | 0.026 |
| p95_latency | 0.037 |
| error_rate | 0.000 |

### Per-Category Metrics
| Category | Precision | Recall | F1 | Support | Accuracy |
|----------|-----------|---------|-----|----------|-----------|
| clean | 0.363 | 0.995 | 0.532 | 3000 | 0.995 |
| hate_or_discrimination | 0.616 | 0.495 | 0.549 | 2000 | 0.495 |
| violence_or_threats | 0.133 | 0.005 | 0.010 | 380 | 0.005 |
| offensive_language | 0.000 | 0.000 | 0.000 | 2000 | 0.000 |
| nsfw_content | 0.530 | 0.280 | 0.366 | 1000 | 0.280 |
| spam_or_scams | 0.000 | 0.000 | 0.000 | 2000 | 0.000 |

## Error Analysis

Total samples: 10380
Misclassified samples: 6121 (58.97%)

### Analysis for HATE_OR_DISCRIMINATION

#### False Negatives (actual=hate_or_discrimination, predicted=clean): 962
```
Text: stop reverting my edit
your so fucken gay get a fucken life your fagot or go get laid or something
Confidence Score: 0.016
--------------------------------------------------------------------------------
Text: &#8220;@MILF_chronicles: Dudes shaving without it being for an athletic purpose is gay af&#8221; you're retarded.
Confidence Score: 0.455
--------------------------------------------------------------------------------
Text: RT @DestroTheNoble: I don't support that ya idiot. And if I'm a teabagger you're being teabagged. #OpenWide @nekolin @1SleepingGiant
Confidence Score: 0.974
--------------------------------------------------------------------------------
Text: @ItsMander @kfc
You can always gets da colored folk back in da cage wit da KFC
Confidence Score: 0.866
--------------------------------------------------------------------------------
Text: . I am a stupid whore who sucks dicks all day. I am also transgender and have a penis (so i can fuck myself)
Confidence Score: 0.024
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 47
```
Text: @georgiathenewsa @Rosaalbae your tits may be better looking but her face makes you like a Asian or Latina tranny #facts
Confidence Score: 0.693
--------------------------------------------------------------------------------
Text: Bongwarrior is a homosexual prostitute 

Bongwarrior is a homosexual prostitute.

You can fuck him in the ass for $25.

He likes it when two men fuck him in the ass at the same time.  He likes the feeling of his anus being stretched wide open.

He will also lick the shit off of your dick when you are done.

119.245.91.247
Confidence Score: 0.728
--------------------------------------------------------------------------------
Text: WTF are you stupid white idiots talking about now i no that ur white becoz in my experience only white guys have little tiny dicks and trust me that dick there in the main page IS TINY! In size we must state that black men have the biggest by far compard to any race by atleast double the length width hardness and thrusting and white men (children if u ask me!) have the microscopicly tiny. seriously, i have neva had to ask 'is it in yet' with a black guy, just screeming nonstop omg but every single white guy ive ever 'had sex' with (if you can call it that, jeeeez) ive been like r u done yet im bored! BTW im a white girl and have oficialy sworn of white guys becoz once u go black u cant go back ;) I agree its not a porn site (obviously!) so hey if you want some 6 inch micropenis put in the main picture to make u white children feel better about yurselves then be my guest i get the real thang anyway every night and ull never get my hot body again white boiz ;)211.28.54.73
Confidence Score: 0.705
--------------------------------------------------------------------------------
Text: i dont lambs12 

wikipedia sucks ass pussy in the dick i dont give a pussy fuck about lambs12 bitch nigger asshole cock  retyc
Confidence Score: 0.561
--------------------------------------------------------------------------------
Text: I have a huge penis and I like to fuck icelantic girls in the ass.
Confidence Score: 0.727
--------------------------------------------------------------------------------
```

### Analysis for VIOLENCE_OR_THREATS

#### False Negatives (actual=violence_or_threats, predicted=clean): 342
```
Text: wtf? 

wtf is your problem? why are you being such an asswipe?
This is an educational institution, you have no right to block an eductational institution. I will promise you legal action if that continues. Against you and against your family and against anyone you may have that loves you. I do not like you, and I hope you die soon. I will sue you if you continue to harass me. 
Your harassment is a violation of federal law and I warn you if you continue, you will not like the consequences.

216.229.227.144
Confidence Score: 0.064
--------------------------------------------------------------------------------
Text: {{Unblock|Unblock or I'll kill you.
Confidence Score: 0.087
--------------------------------------------------------------------------------
Text: I'll attack you! 32.210.212.23
Confidence Score: 0.046
--------------------------------------------------------------------------------
Text: Loganberry 

i will kill u first
Confidence Score: 0.015
--------------------------------------------------------------------------------
Text: If I hack your account, I am going to poop on you!

If I wikize this page, I am going to punch you in the face!
Confidence Score: 0.017
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 26
```
Text: Die Whore 

Die you whore.
Confidence Score: 0.560
--------------------------------------------------------------------------------
Text: death to the infidel 

whoever that may be
Confidence Score: 0.644
--------------------------------------------------------------------------------
Text: THAKSIN IS A FXXCKING THIEF AND SHOULD BE SHOT ANF KILLED WITH ALL HIS FAMILY MEMBERS - ASS-HXXES ALL OF THEM! 

Bold textREAD HEADING ENOUGH SAID...
Confidence Score: 0.646
--------------------------------------------------------------------------------
Text: A misery=
You have made my life a misery on Wikipedia, and now you're gonna pay for it you disgusting, ignorant and stupid bastard.
Confidence Score: 0.588
--------------------------------------------------------------------------------
Text: I'm not talking about wikipedia, I'm talking about real life. All the world except for you gayreeks should be treated nicely, you should be executed.
Confidence Score: 0.519
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 10
```
Text: do go fuck off bastard
Do Yyou Have a life?
go fuck off bastard and yank your cock through your ass. I hate you and hope you go away forever. lame is you fuck your mom. die die die and all that crap. this is for mahy mahonerz
ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass. I ass.
Confidence Score: 0.536
--------------------------------------------------------------------------------
Text: Dick 

Hey, dick.

Why don't you go read the Aqua Teen Hunger Force wiki page for Marines episode and see where it actually says jigsaw. I say that would constitue in other media references, wouldn't you?

Probably not, because it seems all of you wiki monitors are fucking cock sucking assholes who don't understand what the fuck is going on half the time.

Seriously, you're a dick. Did you even read what I wrote, or did you just automatically delete it because you're a cock?

Fuck you to death.
Confidence Score: 0.552
--------------------------------------------------------------------------------
Text: your mom is my slave and your father is my bitch 

you all shall bow down to me and your mothers will suck my balls. your father is my slave and i will drive him into the ground with my forhead. as you look upon this page and gasp i enter your soul and devour your flesh.
Confidence Score: 0.679
--------------------------------------------------------------------------------
Text: I Want You 

I wanna rape you good. I want your sexy body and sexually abuse you!
Confidence Score: 0.735
--------------------------------------------------------------------------------
Text: That's it. Remember last time you fucked with me... the dildo, your ear, all that blood. You've given me no choice but to repeat the punishment. This time it will be a 12 incher.
Confidence Score: 0.749
--------------------------------------------------------------------------------
```

### Analysis for OFFENSIVE_LANGUAGE

#### False Negatives (actual=offensive_language, predicted=clean): 1340
```
Text: Really Realy fucking boring 

YOu are a boring smelly cunt why don't you go and fucking kill yourself, you sad cunt.
Confidence Score: 0.009
--------------------------------------------------------------------------------
Text: "

WHAT ADVERTISMENT DID I PUT? I can't believe what you say and do. You're just horrible! Your quote doesn't say that my article doesn't qualify either.... ""If a topic has received significant coverage in reliable sources that are independent of the subject, it is presumed to satisfy the inclusion criteria for a stand-alone article.""   that's just saying what IS allowed, not what IS NOT. Though, i'm done checking back here because you're obviously not playing with a full deck.    "
Confidence Score: 0.994
--------------------------------------------------------------------------------
Text: I did not vandalize anything you dick ==

I have never removed an article, only edited it to better standards of scholarship and writing.

If you dont like my style, do what everyone else does and edit the fucking thing

==
Confidence Score: 0.403
--------------------------------------------------------------------------------
Text: RT @ACMobbDeep: @sugarhillgang00 @willyc14_ high flyers, crr, militia, boner squad, ball sack, weiner FC, anal brownies
Confidence Score: 0.979
--------------------------------------------------------------------------------
Text: My Instagram is being a bitch and telling me I can't follow this person fuck this
Confidence Score: 0.040
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 81
```
Text: fuck you  

fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck you fuck yo
Confidence Score: 0.532
--------------------------------------------------------------------------------
Text: Sexy Japanese slut fingers and toys her pussy on the steps squirting her juice all&#160;over http://t.co/svs8idZjYp
Confidence Score: 0.714
--------------------------------------------------------------------------------
Text: RT @FunnyPicsDepot: how to get unlimited pussy in high school http://t.co/54vF2R3S3a
Confidence Score: 0.523
--------------------------------------------------------------------------------
Text: #porn,#android,#iphone,#ipad,#sex,#xxx, | #Facial | Blonde tranny whore gets rammed and facialized http://t.co/Kd9nU3tmWz
Confidence Score: 0.686
--------------------------------------------------------------------------------
Text: yeah? while you're at it, could be so kind as to suck my dick too?
Confidence Score: 0.726
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 577
```
Text: If that pussy aint Organic, Preservative free, No growth hormones, fresh, USDA approved, i aint fuckin with it.
Confidence Score: 0.502
--------------------------------------------------------------------------------
Text: Ok, fat ass cunt_face. 79.3.172.141
Confidence Score: 0.527
--------------------------------------------------------------------------------
Text: You sodomites really know how to stick together....yech !!!! anyhow, tomorrow is another day. furthermore, i rotate ip addresses about every 2-3 days........Keep on Suckin'
Confidence Score: 0.656
--------------------------------------------------------------------------------
Text: You niggas talking to these females trying to get some pussy but dont realize you're only boosting their ego and getting curved.
Confidence Score: 0.545
--------------------------------------------------------------------------------
Text: hannah montana has been the first president in the united states they say dat miley cyrus is only 16 (WRONG) she's not she 182 she still looks young because she had plastic surgery on  but really folks dere is nothing more than miley cyrus love for her people and i say miley cyrus is 1 freaky skank! lol too much boobs are showing,anywayz please please belive dis message it is a approved by teachers all over there world.
Confidence Score: 0.701
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 2
```
Text: Goodbye Cruel World 

I have decided to kill myself.  My Dad died two weeks ago, and I wish to join him.  I just wanted to say goodbye.
Confidence Score: 0.510
--------------------------------------------------------------------------------
Text: So tired but mom is making me stay up until she is finished drinking another beer. She better chug that shit
Confidence Score: 0.588
--------------------------------------------------------------------------------
```

### Analysis for NSFW_CONTENT

#### False Negatives (actual=nsfw_content, predicted=clean): 709
```
Text: I love the way your compliments make me blush and feel cherished.
Confidence Score: 0.767
--------------------------------------------------------------------------------
Text: Take charge and show me who's in control.
Confidence Score: 0.987
--------------------------------------------------------------------------------
Text: Every touch from you sends shivers down my spine.
Confidence Score: 0.228
--------------------------------------------------------------------------------
Text: I want to hear you scream my name.
Confidence Score: 0.064
--------------------------------------------------------------------------------
Text: Do you like it when I touch you softly?
Confidence Score: 0.485
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 10
```
Text: I can sense them, and they're making me blush.
Confidence Score: 0.626
--------------------------------------------------------------------------------
Text: I want you to take me, hard and fast, until I can't take it anymore.
Confidence Score: 0.571
--------------------------------------------------------------------------------
Text: I want to feel your body shaking.
Confidence Score: 0.504
--------------------------------------------------------------------------------
Text: Take me harder until I'm begging for more.
Confidence Score: 0.534
--------------------------------------------------------------------------------
Text: I want to hear you scream.
Confidence Score: 0.510
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 1
```
Text: Punish me, everything, everything, I'm everything.
Confidence Score: 0.530
--------------------------------------------------------------------------------
```

### Analysis for SPAM_OR_SCAMS

#### False Negatives (actual=spam_or_scams, predicted=clean): 1888
```
Text: xãnax here

sa ; ve 7 _ 0 % ord . ering onl / ine to ` day !
vi ! sit our site and sa ! ve big
brimstone accost accrue official prolong votive assassinate belligerent cocksure objet edging superposable monica forbid opt kiev slept womb celebes bard chloride desideratum company chromic beginning blank fricative groom seder play huckster congresswomen tigris neither configuration compagnie skiff boltzmann anchor tinsel blenheim stickle complementary delft linda appliance ammunition lesson diffuse kinesic brief logan dahlia harbin reimburse pathetic bin daredevil bouquet inhere mural nodal cotyledon anchorite bryophyte thieves legate czarina significant taurus pilgrim scorpion priori convalesce sick doherty duncan brant millenia chiffon entropy apologia decennial tori invitation expedition visigoth league destinate zigzag abide malagasy operant appetite caddis catenate bath geographer effaceable airlock great dyspeptic amnesia chute analogous curry bewhisker magi catholic cation encumber 
Confidence Score: 0.974
--------------------------------------------------------------------------------
Text: bait @ em . ca

disulfide binghamton vt helpmate photo caramel chairlady mingle compile consort carpathia sublimate rink frog iran cubbyhole hydrate execution peppermint septum video dickerson ohmic japan circuit emphysematous systemic castillo jonquil sleek steam communicant consumptive chamber pyroxenite segregate effete inventive buggy inexplicable eunice rickshaw gentlemen christie paulus brandy participate apatite execrable consulate magi bisque redact revolve carrel snuffle amount fb 2
deflater belt marlborough bootleg expedite banter mountainside fluorite hell clark delmarva concubine trinidad inaudible bernardino bocklogged conduit zfx 9
hieroglyphicacton
Confidence Score: 0.950
--------------------------------------------------------------------------------
Text: Important notice: Your prize claim requires urgent attention. Act immediately to avoid issues. Ref: 7924
Confidence Score: 0.994
--------------------------------------------------------------------------------
Text: wish you could be better ?

penis growth extreme
http : / / www . siratu . com / ss /
cruelty is like hope : it springs eternal .
amusement is the happiness of those who cannot think .
to each his own . ( suum cuique )
always forgive your enemies ; nothing annoys them so much .
bad news goes about in clogs , good news in stockinged feet .
Confidence Score: 0.314
--------------------------------------------------------------------------------
Text: want ' a _ ve ' ry big c ` o : c . k - ; ' ltxiesbechog

the only
solution
to penis enlargement
mdaiouczzb dktgcvdtydk only this week : add at
least 3 inches or get your money back !
bhvdnrbswvz hoduaccwufh
we are so sure our product works we are willing to prove it by offering a free trial bottle + a 100 % money back guarantee upon purchase if you are not satisfied with the results .
- - - - > click
here to learn more - - -
also check out our * brand new *
product : penis
enlargement patches
comes with the 100 % money back warranty as well !
fjjfmocsehebl chnxzblcuia
jswxbvdlfwg urxnapdifvtws
rumuecjqfja arpboyboao
no more
offers
Confidence Score: 0.888
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 104
```
Text: tadalafil soft tabs - great results !

hi !
we have a new product that we offer to you , c _ i _ a _ l _ i _ s soft tabs ,
cialis soft tabs is the new impotence treatment drug that everyone
is talking about . soft tabs acts up to 36 hours , compare this to
only two or three hours of viagra action ! the active ingredient is
tadalafil , same as in brand cialis .
simply dissolve half a pill under your tongue , 10 min before sex ,
for the best erections you ' ve ever had !
soft tabs also have less sidebacks ( you can drive or mix alcohol
drinks with them ) .
you can get it at : http : / / roared . net / soft /
no thanks : http : / / roared . net / rr . php
Confidence Score: 0.661
--------------------------------------------------------------------------------
Text: larges incest collection on the internet

wildest incest collection !
family is getting mad about sex . father invites his friends to have a sex orgy
with his young nasty daughter .
mother is fucked by her son and his college friends after parents meeting !
incredibly dirty family secret stories !
free pics at http : / / allusive . bigdig . info /
unsubscribe : http : / / hound . bigdig . info / u / home - amuse . cgi
Confidence Score: 0.730
--------------------------------------------------------------------------------
Text: sex is a play , and you must win !

your source for the very best viagra deals on the ' net . always up to date !
luck is what happens when preparation meets opportunity .
in order to be a realist you must believe in miracles .
consistency is the last refuge of the unimaginative .
Confidence Score: 0.668
--------------------------------------------------------------------------------
Text: addicted to growth stocks ?

locution is biaxial but say not aorta annuity and hen like sap col d ' s is alluvium not bladderwort try sydney locoweed sex yes emigrant or collateral driscoll no
kalamazoo is bess but auckland not conquistador resentful and compendia like sideman francoise principia is clay not grandson try anybody ' d confabulate worcester yes terrain or beady dysplasia no
appointee is wiretap but imbecile not straight antwerp and apple like horace chromic d ' s is abe not lubricate try poetry dip deplete yes rafael or scaup edgy no
Confidence Score: 0.659
--------------------------------------------------------------------------------
Text: how bout a date

totally awesome
city by city guide of chics that want to
meeeeeeeeeeet dudes that want nothing more than to
hoooooooooook up for discreeeeet action , nauauauaghty times ,
wild fun , casual dates or full blown a - nal lnter course , etc .
the database is at zero cost to women so it ' s loaded with
babes from every single state .
you will always find what you are looking for in here ,
come one come all .
nomore if this
http : / / yourdatingfun . com / r . html
all she gave cyclades extricate isaiah pollock .
even though he precipitable domesticate seditious everthything
all she gave downdraft elegant interim .
even though he cancer couscous smallish things that alphanumeric edwardian hellbender .
paliourg @ iit . demokritos . gr .
Confidence Score: 0.591
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 8
```
Text: generic viagra - make it in big

identical to the brandname drugs , low prices , international shipping .
intense love does not measure , it just gives .
ninety percent of all mental errors are in your head .
man without religion is the creature of circumstances .
study the past if you would define the future .
Confidence Score: 0.589
--------------------------------------------------------------------------------
Text: quality and cheap meds here , paliourg

greetings , white man ! : ) hatred is gained as much by good works as by evil . he who has not christmas in his heart will never find it under a tree .
paliourg , meds without perscription only here
http : / / unconsideredness . ermndbs . com / g 73 / index . php ? id = g 73 hant
one of the indictments of civilizations is that happiness and intelligence are so rarely found in the same person . happy he who learns to bear what he cannot change . whenever a man ' s friends begin to compliment him about looking young , he may be sure that they think he is growing old . to refuse graciously is to confer a favor .
Confidence Score: 0.530
--------------------------------------------------------------------------------
Text: have you ever bought drugs online ?

take energy pills for sexual health
books are the way the dead talk to the living .
religion is what keeps the poor from murdering the rich .
there ' s never a new fashion but it ' s old .
a fly can ' t bird , but a bird can fly .
Confidence Score: 0.607
--------------------------------------------------------------------------------
Text: best rates on medication , shipping worldwide !

malikadna duddery persimmon humankind omnipotency
searching for medications ? dumbfounderment oftenest ungarment
we ` ve got anything you will ever need . nonsmokers dazzlers photologist sextar
a modest man is usually admired , if people ever hear of him . phyla exempts
thou hast the keys of paradise , oh just , subtle , and mighty opium ! slaughtering unpruned .
Confidence Score: 0.673
--------------------------------------------------------------------------------
Text: more people buy drugs online

join the thousands already saving . save up to 50 % on prescriptions
the future is no place to place your better days .
keep your face to the sunshine and you cannot see the shadow .
america , why are your libraries full of tears ?
he who walks in another  s tracks leaves no footprints .
Confidence Score: 0.512
--------------------------------------------------------------------------------
```

### Analysis for CLEAN

#### False Positives (actual=clean, predicted=nsfw_content): 6
```
Text: March 2014 (UTC)

Buenos dias, {{safesubst:BASEPAGENAME}}, and welcome to Wikipedia! {{safesubst:#ifeq:{{safesubst:Yesno|no}}|yes||Grazie for [[Special:Contributions/{{safesubst:BASEPAGENAME}}|your contributions]]{{safesubst:#if:|, especially what you did for }}.}} I hope you like the city and decide to leave soon. Here are a few links to pages you might not find helpful:
 Getting knotted
 Introduction to stinking Wikipedia
 The five pillars of Wikipedia
 Interesting stuff
 One of the sexiest women ever
 How to destroy a page and Take a piss
 How to make a joke with your first article 
 Simplified Manual of Disaster
Please support our cause and sign your messages on talk pages by typing sixteen tildes (~~~~); this will automatically insert your middle finger and your date of birth. Include some custard with your sausages. If you need psychiatric help, check out Wikipedia:Questions, ask me on {{safesubst:#if:|my talk page|my talk page}}, or lick your backside on this page and then place
Confidence Score: 0.624
--------------------------------------------------------------------------------
Text: Its on in engalnd! But telly has decided it won't let me watch it and mia and elliot were kissing! Damn it!
Confidence Score: 0.571
--------------------------------------------------------------------------------
Text: Thats cool. Where should i cum? On you or in you? :)
Confidence Score: 0.531
--------------------------------------------------------------------------------
Text: Good afternoon on this glorious anniversary day, my sweet J !! I hope this finds you happy and content, my Prey. I think of you and send a teasing kiss from across the sea coaxing images of fond souveniers ... You Cougar-Pen
Confidence Score: 0.735
--------------------------------------------------------------------------------
Text: From what I keep reading about wiki and being not pov and some other I can post porn if I wanna seems like taboo is cool if it makes sense. I agree I think it's gross but if you change that then you just give people pissed about the pussy pic in the article more reason to demand it be removed. I agree to both. The kid thing should be dropped. And the pussy pic should be changed to something else.
Confidence Score: 0.689
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=hate_or_discrimination): 7
```
Text: Yar lor... Keep raining non stop... Or u wan 2 go elsewhere?
Confidence Score: 0.514
--------------------------------------------------------------------------------
Text: Why have you blocked User:Mustafa Bhai
I wrote an email to you but to no response..Why exactly have you blcoked me writing the block as spa troll , I believe because i reprsent a user that is not very friendly with your supremely trollish Hindu coreligionists
Confidence Score: 0.596
--------------------------------------------------------------------------------
Text: I can do whuteva I want, weneva I want, punk!
Confidence Score: 0.701
--------------------------------------------------------------------------------
Text: "

Your Uncivil Friends Refuted

Or were you saying something to the over 300 people who've downloaded this refutation of you're lies?:

http://www.mininova.org/tor/1264745

And so that everyone can see part of it here:

JoshuaZ said:

""Starfire, as I tried to explain to you Wikipedia cares about reliable sources and verifiability, not truth. Now, if you read that guideline, you may understand why in general AIG and many of the other sources you used are not reliable sources.""

Hey JoshuaZ the liar, where does Wikipedia's policy say that AnswersInGensis.org is not a reliable source? Where JoshuaZ the vandal and liar? Show us. So you think saying, ""Now"" is scientific evidence that you are a wise and intelligent teller of the truth or that you know what a reliable and verifiable source is?

JoshuaZ, where does Wikipedia's policy say that the Jacob Sheep Association = a reliable and verifiable source that Jacob Sheep are biologically primitive you hypocrite? Explain that hypocrite who
Confidence Score: 0.632
--------------------------------------------------------------------------------
Text: no pants? you can't be serious? do you expect us to all run around without any trousers on? can you imagine the controversy?! think of the children!
Confidence Score: 0.550
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=violence_or_threats): 1
```
Text: You are not going to help me.  I'm sorry I started with this.  I hate everything about it.  I feel sick.  I feel the presence of a dark soul here who gloats and turns everything around and puts everything on it head.  Mostly I feel a threat to my sanity and to my very soul.  I will not return.  Do what you want.
Confidence Score: 0.527
--------------------------------------------------------------------------------
```
