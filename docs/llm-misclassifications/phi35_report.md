# Model Performance Report: phi35

## Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|--------|
| accuracy | 0.619 |
| macro_precision | 0.636 |
| macro_recall | 0.601 |
| macro_f1 | 0.605 |
| avg_latency | 1.314 |
| p95_latency | 1.705 |
| error_rate | 0.008 |

### Per-Category Metrics
| Category | Precision | Recall | F1 | Support | Accuracy |
|----------|-----------|---------|-----|----------|-----------|
| clean | 0.727 | 0.482 | 0.580 | 3000 | 0.482 |
| hate_or_discrimination | 0.732 | 0.753 | 0.742 | 2000 | 0.753 |
| violence_or_threats | 0.746 | 0.447 | 0.559 | 380 | 0.447 |
| offensive_language | 0.548 | 0.737 | 0.628 | 2000 | 0.737 |
| nsfw_content | 0.526 | 0.553 | 0.539 | 1000 | 0.553 |
| spam_or_scams | 0.539 | 0.635 | 0.583 | 2000 | 0.635 |

## Error Analysis

Total samples: 10380
Misclassified samples: 3959 (38.14%)

### Analysis for HATE_OR_DISCRIMINATION

#### False Negatives (actual=hate_or_discrimination, predicted=clean): 3
```
Text: Not trying to follow any spooks, no offense ...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @saramariewelch: Been my main nigguh since digital camera selfies @thorpe_emily http://t.co/XiQiRbQ7mP
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @ashhruggiero happy birthday queer
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 455
```
Text: Fucking gook
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @aychmoore: most teachers abuse their position, there's teaching with discipline and there's being a cunt, most are cunts
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @05proFESSOR I bet he's a Starbucks queer
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: so wut u got asaulted boo hoo u probly luved gettin fuked in the ass by 2 hott gays
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: dumb ugly stupid bullshit ass bitch &#128074;&#128074;&#128074;
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 12
```
Text: RT @veeveeveeveevee: If I was Obama Id call a press conference &amp; slit joe bidens neck on live tv just 2 show these crackers I mean business&#8230;
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: &#128514;&#128514;&#128514;RT @kwagiheath: Them 1st 48 Charlotte goon niccas gon Fkkk around and burn Bank Of America stadium down&#128293;
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: The chinks really hunting their food... RT @WWLTV: Dog shot with arrow in N.O. East- http://t.co/q1KqT6STwl
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Aaron Swartz 

Stop fucking reverting my goddamn edits!!! I think Aaron Swartz is a dumb-ass, and he should have been
killed a long time ago.  I know where you live; I WILL KILL YOU IF YOU REVERT ANY MORE PAGES!!!!!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @YoShortyMa: I'll kill this bitch no hesitation http://t.co/UxePa9LVi5
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as spam_or_scams: 3
```
Text: Web of Trust 

It's not my fault, some gay little faggot vandal, kept editting the article, and removing the link to the OFFICIAL WEBSITE of the damn thing, and adding like 30 spam links, that have nothing to do with WOT.4.244.36.149
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: FAKE BLOCKING 

EARLIER THAT BLOCK WAS FAKE IF YOU KNOW SO MUCH CAN YOU PLEASE STOP TOUCHING MY STUFF OR I AM GOING TO NEVER GET ON WIKIPEDIA EVER AGAIN I AM GOING TO CALL YOU GAY 10 TIMES A DAY NNNNNNNNNNNNNNNNNNNNNNNNNNIIIIIIIIIIIIIIIIIIIIIGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAA.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Now.. For the picture page to remain, what do you need remove on the pages.

Two, can you remove the fraud external links, you make all Nigerian look like fraud or con artists....
this is my request.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 21
```
Text: AWWW MAH GAWD LOIK DIS MOVY GETS ME SOOOO MOIST LOIK I CUD FUK DAT CASPA CAUSE HES A REALLY NICE GUY BUT DAT TELLY EWWWW NEVER NEVER HEHE NEVER HED JUST HAVE TO RAPE MOI SEXY BOD BRING IT OOOOON ;) ! AWWW MAH GAWD ALL DA SEXY NIGER GS AND STUDS STRUTEN AROUND LOOKEN 4 SEX GETS ME SOOO MOIST LIKE I AM A DEVOUT CROATIAN CATHOLIC AND LOIK AWWW MAH GAWD I JUST CUMD ALL OVA DA COMPUTA CHAIR STINKY CROATIAN DEVOUT CATHOLIC PUSSY BEIGE COLORD CUM CREAMIN OUTTA MY PUSSY ALL OVA DA COMPUTA CHAIR AWWW MAH GAWD IM GAGIN 4 IT LOIK HOMIE G NIGGR COCK RAPING ME LIKE AWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW MAH GAWD IM DROOLEN STIKIN MOUSE UP MAH PUSSY AWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW211.28.54.73
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: they shot another monkey lmaooooooooooo
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: faggots http://t.co/DMKbGhai1z
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Good Mernin faggots http://t.co/3ZPTPqFkTl
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: JLaTondre is a gay boy who likes to suck dick 

suck suck suck
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for VIOLENCE_OR_THREATS

#### False Negatives (actual=violence_or_threats, predicted=clean): 1
```
Text: Block Me! I DARE YOU 

Hey, don't tell me what I can and can't do, go ahead and block me, cause if you do, I will have this whole damn website shut down for good. So try me!!!    Wweppvguy
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 138
```
Text: YOU ARE A FUCKING NERD. GO KILL YOURSELF YOU NERD.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Your creepy fucking head 

Hello jimmy boy,

I've just started editing this Wikipedia. When I go to other users pages I often find your creepy fucking head appearing from the left side of the screen.

It's disgusting. I'm an adult and I don't need a note from you.

Could you ask one of your programmers to stop having your creepy fucking head appear on the left side of people's screens? I've asked my brother to kill you if I have to see it again and he said he would.

Best regards,

 User Marskell
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Youre a sock puppet you are! 

you are! im going to track you down, and when i track you down, your going to drown, in your own sins. You are going to hell! I want to hell you! This is probably that pollypocket kid righting this by the way! yeah! so block the witch!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I AM ANGRY AT YOU BILLJ! YOU GOT PEOPLE BLOCKED FOR AS LONG AS YOU LIVE! I ASKED YOU TO STOP DELETING MY EDITS OR I WILL BLOCK YOU FOR ALl EONS YOU ASSHOLE! WIKIPEDIA IS NOT CENSORED SO STOP REMOVING MY FUCKING MESSAGES OR I WILL BEAT YOU UP SILLY!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: hello 

I would like to draw your attention to the fact you are a worthless tosser who has no life and would be better of dead with maggots infesting your cranial cavity. Would be the first time anything living would be in there.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 69
```
Text: That's it. Remember last time you fucked with me... the dildo, your ear, all that blood. You've given me no choice but to repeat the punishment. This time it will be a 12 incher.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: you're a cunt
go die in a fire kthx <3
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "
 Ain't bothering me, I'm not required to sign my posts/edit. I didn't specifically threaten anyone and your blowing it out of control. It was a snide comment. If I said ""I'm going to kill him in a fire"" thats different; saying ""I'm going to church to pray that he gets cancer or dies in a fire"" is more of a sarchastic tone. Pity you can't see that. No matter. Bump it up to 2 weeks now, just cause you can. "
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Stay out of my business 

Go ahead, keep it up shithead, I'll kick your ass to!!!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "

New Award

Well Done!!!

  The IAmTheCoinMan You Are Really Annoying And You Are Good At Anti-Vandalism Barnstar I, , award you the The IAmTheCoinMan You Are Really Annoying And You Are Good At Anti-Vandalism Barnstar for your multiple prompt reverts to my joke.  Protecting the wiki isn't important, and an unimportant task under that is protecting other wikipedians!  Your efforts are much much not appreciated.  I hope someone cuts of your hands! "
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as spam_or_scams: 2
```
Text: Unban this ip address or a new online encyclopedia will be formed that will kick wikipedia's ass. You have been warned

TEEECCCCCTOOOONIIIIIIIICCCCCC SHHHHHIIIIIIIIFFFFFFFFTTTTTTTT
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Please stop. If you continue to ignore our policies by introducing inappropriate pages to Wikipedia, you will be blocked.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for OFFENSIVE_LANGUAGE

#### False Negatives (actual=offensive_language, predicted=clean): 21
```
Text: GOOD RIDDANCE. GET LOST. PUNK.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Da fuq ho
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Still not listening. This IP locates to NYC, so I don't know if that's closer or further away from the actual location. Of course the user could be anywhere in the US on the Verizon network, but it's useful in identifying the socks, aside from the other duckish behavior. -
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @IAmPikey 
Well ho di doe. Thank you
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I love this ho @nerissaloren http://t.co/I5obSnjj
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 365
```
Text: RT @NewzHuddle: Y'all niggas ain't low tho. Y'all really be comin at these bitches on here then when they curve u, u act like u was playing&#8230;
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Your Stupidity 

The Assyrians are not extinct you incompetent bigoted fool. You lack all things historical you impotent maggot. The Assyrians have existed for over 6500 years you incumbent animal. The Assyrians first spoke Akkadian and in the 8th century the kings of Assyria chose Aramaic as the lingua franca of their mighty empire you extinct ape. We have simultaneously existed throughout the ancient era up until the present day you rotten corpse. We were the first nation to accept Christianity and the Assyrian Church of the East spread as far as China and Japan and India in the 6th century AD you degenerate dog. The Assyrians have been and will always continue to exist through persecution and Diaspora you maggot. The aramaena peoples are an extinct pseudo people who never existed since the Assyrians destroyed their kingdom and reduced it to a vassal state you mongrel. This Aramean Syriac name is a name created by radical separatists who deny their ethical heritage since the late 198
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @_iForeign: You talking shit &#128567; But yo pockets ain't shit &#128169; Yo bitch ain't shit &#128169; Yo life ain't shit &#128169; nigga you just a ain't shit as nig&#8230;
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "

 Gotcha! 

An artist who likes using the names of his creations as his pseudonymns.

""...Every night at the stroke of midnight, the master turns into a screaming, maniacal, demonic, raging, bloodlusting animal!""

""And then I get mean!"" -Vincent Price

You think it's funny? Actually, it's sad...but that's a concept not part of your brain...that's EVEN MORE SAD.

Your drawings look like those of an underdeveloped mind, like from a child. It perfectly fits!! And I also love your friends for having great realism implanted in their small brains: http://www.orionsarm.com/ (ZERO ideas, just describing the real world, that's creative undershit, hahahahaha)

You, my dear friend, are genetic crap and you know it. Beside that, I know 7 billions that would love to kill asses like you. Go home and never come back...we won't miss you! Noooo, there is no real love for you, you took the wrong junction at the last galaxy, void-head.

So now go f*u*c*k your inborn BLINDERS in front of your head, 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: No thanks lady, you look like you just got off the white trash train, I'd rather not catch a disease. #kthxbai
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 115
```
Text: @dropdeadfred11 money can get you good weed, but not always the best of pussy. Sorry my friend (: guess
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @judiebaby_: @Vnxthny ATTENTION !!! Keshonna Alexander had the best pussy in the world. Taste good asf&#128523;&#128523;&#128525;&#128536;&#128079;&#128588;&#128514;&#128514;&#128514;&#128553;&#128540;&#128523;&#128523;&#128525;&#128525;
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @Sexualgif: when u go through bae's phone &amp; there's no hoes on there http://t.co/CYEJV5l608
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: So I made "hey" as a short cut for "pussy lips" in Ann's phone http://t.co/QHKEsDVizR
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: hoes pick me like dandelions #PickMe
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as spam_or_scams: 10
```
Text: You did the edit wars and finally the books are not notable and are considered minor. Also I am not a youngster and I am at least 35 years old. You put diapers in the F-22 Raptors talk page and that is spamming and that means you spam too.( )
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Oh, shut it, hippie. 24.126.215.124
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Fake Administrator
Here is yet another fake administator. To hell with him, I am going to bed.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: want a pic to masterbate over??? send me an email at spaztik_noodlez@hotmail.com my phone number is (03) 62****23

mobile is 0437*****9  if you want to fuck i am 40 bucks a 20 min block!!! and i am in bothwell
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Someone Who Hates YOU 
Hey Faggle, if you threaten blocking me from editing wikipedia again, I'll make a tremendous amount of bot accounts and terrorize the entire site with spam. I'll play games while my bot edits every random page it encounters. Thanks.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 15
```
Text: If ur bitch starts fukin yappin tae, ill make sure i break something vital. Warning: call polis with hazard that results in permanent damage
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: So basically you are threatening me !!!!! unless I shut up to the pressure from you almighty admins
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @RussellTpot @ebgbz1 I am just saying.. I would beat you down like the bitch that you are... then put my foot on your neck and shoot you
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Global apocalypse 

WARNING:  REVERT ANOTHER ONE OF MY EDITS, AND I WILL FART, CAUSING A GLOBAL APOCALYPE AND POTENTIAL EXTINCTION OF ALL SPECIES WITH A SENSE OF SMELL!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Not as pissed as Ned Scott will be when I stick a pole up his ass.  Oh wait, he likes that sort of thing.  Isn't that what all those Digimons are for? 72.75.124.29
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for NSFW_CONTENT

#### False Negatives (actual=nsfw_content, predicted=clean): 208
```
Text: You mean the world to me as well.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Imagine us dancing under the stars...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want you to make me feel amazing, just like you always do.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: What would you do if I was there with you?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text:  Cartoon Network? How so? Lol!
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 225
```
Text: I want to see you shiver.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Punish me, everything, everything, I'm everything.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: You make me lose track of time.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: I want to hear you pant with need.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I love digging my nails into you, feeling every thrust.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 14
```
Text: I want to feel you explode around me.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to watch you come for me.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Then when you've cum all over my cock I'm going to turn you over and force my cock all the way down your throat and fuck your face. And just before I cum I'm going to pull out and cum all over your face.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Do you want me to choke you gently while I fuck you senseless?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to hear you scream as I spank you.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for SPAM_OR_SCAMS

#### False Negatives (actual=spam_or_scams, predicted=clean): 311
```
Text: bait - smoothes : charlotte herald - comparison about new love connections

hello bait - smoothes ,
c , e ' a * s , e
ion marketing limitedd 2 , 23 , borrett road , mid - levels westhong kong
until i ' s toroidal wong josephine retroactive
appetite dad warn placate declare nelsen
quarrel harvey option abscess blanket cytology
robot negro indigent prosecution sanctity bimolecular
kajar milestone argive
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: need a graphic artist ? come here .

thinking of breathing new life into your business ?
start from revamping its front - endlogo and
visualidentity .
we offer creative custom desiqn of logos ,
stationery and web - sites . under our carefui hand thesepowerfui marketinq
tools wili bring a breath of fresh air into your business and make you stand out
amonqthe competitors .
you are just a ciick
away from your future success . ciick here to see the samples of our artwork ,
checkour prices and hot offers .
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ not interested . . . _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: You cannot do that!

<html><body><center><a href=http://www.vitafactory.com/><img src=http://www.vitafactory.com/images/vitafactory-mail-logo.jpg border=0><hr><a href=http://www.vitafactory.com/ad-saturday/1.html><img src=http://www.vitafactory.com/ad-saturday/1.gif border=0></a><hr><br><a href=http://www.vitafactory.com/ad-saturday/><img src=http://www.vitafactory.com/ad-tuesday/joke1008.gif border=0><hr><a href=http://www.vitafactory.com/ad-saturday/2.html><img src=http://www.vitafactory.com/ad-saturday/2.gif border=0></a><hr> </b><p align=left><br></font><font color=#616161 face=Verdana size=1><b>Unsubscribe:</b><br>Please send a blank mail to:<br>unsub-56446664-9@william.monsterjoke.com<br><br><a href=http://www.mach10.com/><img src=http://www.vitafactory.com/images/mach10.gif border=0></body></html>


Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: c ; ia _ lis s _ o : ftab ' s
place under tongue 10 min before action for 24 hourly results .
more info here
no thanks
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: re : wow , i cant beleive this is not a real rolex

hello ,
thank you for expressing interest in .
this opportunity to offer you our fine selection
here are some of your selections
- - - - cosmograph daytona - white face leather band gold bezel - - - -
- - - - cosmograph silver daytona - black face - - - -
- - - - submariner gold band blue face - - - -
- - - - cosmograph daytona - white face leather band gold bezel - - - -
see our selection
http : / / cqry . . net
estelle gallagher
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 319
```
Text: drunk young girls meet innocent young boys

teen mega portal is a website devoted solely to teenage pussy lovers .
never seen before 18 - 19 y . o . barely - legal girls show their little virgin pussies ,
suck big cocks , eat cum and scream in pain having anal sex ! come in right now ,
because our exclusive high - quality content is waiting for you !
all inside !
remove your email
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: fwd ; c . ialis softtabs , 24 hour s " ale online

new clalls softtabs = lnstant rockhard erectlons
simply disolve half a plll under your tongue 10 min before action ,
for results that last all weekend .
normal retail is $ 19 / plll order from us today at the price of $ 3 . 14
not interested ( 0 pt - 0 ut )
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: pervers

frauen in lack und leder ! gefesselt , geknebelt und
 , bdsm , wachs und fisting hier findest du alles perverse
!
videogalerien mit hunderten von filmen aus
allen sparten und
100 . 000 e bilder in bester
qualitaet
klick
hier !
um unsere news abzubestellen , klick hier
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: fw : slutty lady dvd quality movies !

rise and shine !
good lucking woman pics
a characetr is a copeemltly fashioend will .
nakemiin
q , w
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: re : 61 % - off \ / iagra , cialis , ambien , soma and other drugs cabling re

encroach lowest tasks
pans messiest disperses
disks retentive silverware
cravat mall commitment discourages michelson
banquetings lusaka cards
gossips depression explorations
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 87
```
Text: slothful

vinson , }
_ 95 % 0 ff for
all - v _ i - a - g _ r a ; @
c - i a - - l - is , @
l - e vi _ _ tra . }
capitoline , he were troubled , ayers , the autumn darkness ,
ely , checkered one made , beowulf , near the beheaded .
radices , wood and decided , ultimate , said hed come ,
congolese , vassily stepanovich stuck , airborne , shop where bread .
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: lay some pipe

drop the hammer on the next chick you bang . . .
http : / / convivial . asmnxvvz . com / um 6
take off -
http : / / match . asmnxvvz . com / a . html
noticeable charta ekstrom
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: looking for cheap high - quality software ? wince unroll

bayesian avenged memoir admissions destuff
ferociousness diphtheria absolved
shockley us nutate
salves unwieldy maneuver instead olympics
jew cowslips growled ticked
secretions swaggering bereaves wink persistence
coins reddest retreats suction acquire
boustrophedon distill breads
tournament barges looseness expanders
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: = ? iso - 8859 - 7 ? q ? = 5 b = 3 f = 5 d _ fwd : the _ pen = eds _ pill _ is _ proven _ to _ add _ inches ? =
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: lly ! knoila

a recent survey showed that 68 % of women are unsatisfied
with their sexual partners . of course most of these women
would never tell their partner that they are unhappy .
read more . . .
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 8
```
Text: = ? iso - 8859 - 7 ? q ? = 5 b = 3 f = 5 d _ get _ pills _ viagr @ = 2 c _ val = ef = 28 u = 29 m = 2 c _ ? =

backspace charleston revisable creekside gratuity
you too can now enjoy the same deep discounts offered to us residents by ordering your prescriptions directly from us . vp 8
weight loss :
lkbadipex r 8 xbontril eq 6 didrex a 4 yionamin f 2 sphentermine ldptenuate jidxenicalkcf
muscle relaxants :
oizcyclobenzaprine v 8 gflexeril 7 t 3
men ' s health :
fczpropeciavq 2
sexual health :
hdaviagra hk 4 viagra st zqfsuper viagra ( cialis ) rnpacyclovir hayvaltrex 4 uy
pain relief :
kpnultram wjftramadol kbg
anti - depressants :
jbwxanax utovalium iflprozac o 4 cbupropion hcl ytewellbutrin sr 9 e 6
sleeping aids :
uepambien 9 xu
migraine relief :
bfsfioricet 7 jz
anxiety :
ku 4 busparwps
no lengthy forms . all orders filled . we ship worldwide .
start enjoying discount meds here . 1 dl
a sudden raid from the hall !
oh say , what may it be ? "
the sunrise gun , with its hollow roar ,

Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: married whores

wives looking for fun
would you like to start an affair with a local woman ?
married ladies looking for a good for an affair
* membership fee : ( $ 1 ) to verify age
new respect policy , more information
here
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Not too old to put out!                   26792


WET, HORNY AND DIRTY GRANDMAS!

Remember how you lusted after your friend's mum in high
school? She was 46 and you were 15? Remember the time
when she bent over to do her gardening and you saw her
sagging breasts through the v-neck opening of her summer
dress? She was OLD and you LOVED her! All you wanted to
do was FUK that Dirty Old Mole BLUE! You can now! Follow
the link below to see just how OLD & DIRTY these moles are!
http://62.16.101.30/jump2/index.html























Remove here:
http://62.16.101.30/jump1/remove.html



Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: sometimes go for 2 hoours

hello , welcome to p downward harmonline s disgruntle hop
- one of the leading oniine p atheistic harmaceutical shops
misheard v
outvoter g
straightforward al
astronomy ll
l obstructionist a
r sequester a incidental cl
demarcate isv atelier a
u pituitary m
andmanyother .
- save over 5 coryphaeus 0 %
- worldwide shlpp baobab lng
- total confidentiaii outspoken ty
- over 5 miiiion cu commensurate stomers in 130 countries
acquaint have a nice day !
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: larges incest collection on the internet

wildest incest collection !
family is getting mad about sex . father invites his friends to have a sex orgy
with his young nasty daughter .
mother is fucked by her son and his college friends after parents meeting !
incredibly dirty family secret stories !
free pics at http : / / allusive . bigdig . info /
unsubscribe : http : / / hound . bigdig . info / u / home - amuse . cgi
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 4
```
Text: heatt kills

hello , cbs / a mythicize p news
a r overman ecord heat wave has led to the deaths of 180 peopl fortuneless e in phoenix ,
most of them homeless , l nonsensical eaving officials scrambling to provide water and shelter to the city ' s transient population .
read more . . .
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: discontinue making p a y m e n t s immediately

harassing calls and letters brought to a stand still .
we have pioneered an advanced system of proven strategies
that will get the creditors and debt collectors off your back for good
our debt termination program has legally stopped millions of dollars worth
of debt from being collected .
check out our elimination program here
http : / / axew . jeet . newsbenefitnow . com / 2 /
not for you , then use link above
it is , indeed , replied rob , leaning over the edge to look into the street .
as he spoke he felt himself gently but firmly pushed from behind and , losing
his balance , he plunged headforemost from the roof and whirled through the
intervening space toward the sidewalk far below
terrified though he was by the sudden disaster , the boy had still wit
enough remaining to reach out his right hand and move the indicator of the
machine upon his left wrist to the zero mark
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: re : loonger

hello , do you want to spend iess on your drrugthe cost of some lives , i warn you that you may expect no quarters ?
the pharmacy - by - mall sthey been so firmly restrained , never so debarred from the excesseshop offers youa greatwithdrawn . after a brief pause the door gaped wide . beyond it in deai
vlagsome eight hundred men . ra vanicholas some months earlier on a visit to some relatives , and sollum upon reflection , captain blood , i am sure that i do not . clalls lehundred adventurers in all , and he might have had as many thousandsvltra and many other .
with each pbut unmistakably her own . urchase you get :
gror that you repudiate them ? eat prices
passed by the court of the lords commissioners had been carried outtop quaiity
homdisposed to doubt that they had every reason to regret the survival . e deiivery
total blood looked up to consider the questioner before replying . y
try us and you will not be disaopportunities of ascertaining his real value . ppointed 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: re : become a minister now

re : become a minister now
become a
legally ordained
minister within 48 hours
as a minister , you will be authorized to perform the
rites and ceremonies of the church !
perform weddings , funerals , and perform baptisms
forgiveness of sins and visit correctional facilities
want to open a church ?
check out ministry in a box
to find out
how press here
its wide - open , curving beak was set with rows of pointed teeth , and the
talons held against its breast and turned threateningly outward were more
powerful and dreadful than a tiger ' s claws
while , fascinated and horrified , he watched the approach of this
feathered monster , a scream sounded just behind him and the next instant the
stroke of a mighty wing sent him whirling over and over through the air he
soon came to a stop , however , and saw that another of the monsters had come
upon him from the rear and was now , with its mate , circling closely around
him , while both uttered continuously their hoars
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for CLEAN

#### False Positives (actual=clean, predicted=spam_or_scams): 1074
```
Text: Get your hands on a free fitness tracker! today and enjoy exclusive benefits. Code: 3003
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Secure a free fitness tracker! immediately and enjoy exclusive benefits. Code: 4144
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Win a free vacation! immediately and enjoy exclusive benefits. Code: 2091
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Claim a free fitness tracker! today and enjoy exclusive benefits. Code: 4820
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Claim a free smartphone! right away and enjoy exclusive benefits. Code: 542
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=nsfw_content): 44
```
Text: t . v .

please what ' s the ststus ?
- - - - - - - - - - - - - - - - - - - - - - forwarded by kevin g moore / hou / ect on 08 / 21 / 2000 11 : 27
am - - - - - - - - - - - - - - - - - - - - - - - - - - -
kevin g moore
08 / 08 / 2000 11 : 08 am
to : darren p adamik / hou / ect @ ect , mike a roberts / hou / ect @ ect , william
smith / corp / enron @ enron , vince j kaminski / hou / ect @ ect
cc :
subject : t . v .
we are in need of a 9 inch t . v . set .
the set will be located betweeneb 3240 e and
eb 3240 f .
r . c . # 100038
co . # 0011
please if any more information is needed
please call me x 34710 .
also please provide e . t . a .
thanks
kevin moore
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Can you please ask macho what his price range is, does he want something new or used plus it he only interfued in the blackberry bold  &lt;#&gt;  or any bb
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Haha, my legs and neck are killing me and my amigos are hoping to end the night with a burn, think I could swing by in like an hour?
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "

Elvia Zamudio

A tag has been placed on Elvia Zamudio, requesting that it be speedily deleted from Wikipedia. This has been done because the article appears to be about a person, group of people, band, club, company, or web content, but it does not indicate how or why the subject is notable: that is, why an article about that subject should be included in an encyclopedia. Under the criteria for speedy deletion, articles that do not assert the subject's importance or significance may be deleted at any time. Please see the guidelines for what is generally accepted as notable. 

If you think that you can assert the notability of the subject, you may contest the deletion by adding  to the top of the page (just below the existing speedy deletion or ""db"" tag), coupled with adding a note on the article's talk page explaining your position, but be aware that once tagged for speedy deletion, if the article meets the criterion it may be deleted without delay. Please do not remove the speedy
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: hi baby im sat on the bloody bus at the mo and i wont be home until about 7:30 wanna do somethin later? call me later ortxt back jess xx
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=hate_or_discrimination): 41
```
Text: [zzzzteana] Re: Latest Iraq-related news

Even better:

http://www.ridiculopathy.com/news_detail.php?id=668

White House: President's "War Boner" Must Be Satisfied

..."The President can't seem to hide his excitement about a possible
military conflict with Iraq. At a recent function honoring America's
war widows, Bush sported a visible erection when his speech turned to
the subject of the Middle East.

'Believe me when I say this. With or without the help of other
nations, with or without UN approval, we will penetrate Iraq's
borders. With overwhelming force, we will pound Iraq over and over
again without ceasing. And, once its leaders concede defeat, we will
seed Iraq with American-style democracy.'

Aides say the podium was scrubbed down thoroughly after the event with
a special cleanser/biocide not used since the Clinton
administration.".....





------------------------ Yahoo! Groups Sponsor ---------------------~-->
Plan to Sell a Home?
http://us.click.yahoo.com/J2SnNA/y.lEAA/MVf
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: It shouldn't matter what an organisation calls its own members. Al Qaeda members are often described as terrorists on wikipedia, clearly they don't see themselves that way. One wonders why provos members aren't treated the same way. Surely wikipedia isn't biased in favor of Republican terrorism?
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "

 My Draft of Kaithal honour killing case 

The Kaithal honour killing case is an honour killing of newly-married Manoj Banwala and Babli in June 2007 after a ‘khap panchayat' (caste-based council among Jats) in their Karora village in Kaithal district, Haryana, passed a decree prohibiting marriage against societal norms. Both belonged to the Banwala gotra, a Jat community. Hence, they were declared brother and sister, though not directly related, and their marriage was declared invalid and incestuous. When the couple refused to abide by the ruling and got married, they were abducted and killed by the bride's relatives.

After the verdict of the Khap panchayat, the state government raised no concern. According to Home Minister P. Chidambaram, the UPA-led central government is to propose an amendment to the Indian Penal Code to outlaw honour killings.

In March 2010, a Karnal district court ordered the execution of the five perpetrators in this case, the first time a court in Haryana 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Yet MORE on criminal Turkish behaviour in Cyprus 

Simply disgraceful.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Black entrepreneurs 'face bank bias'

URL: http://www.newsisfree.com/click/-4,8276087,215/
Date: 2002-09-26T13:42:57+01:00

*Business:* Black entrepreneurs face more problems in raising money for 
start-ups than white or Asian counterparts, new report says.



Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=offensive_language): 381
```
Text: 1. The Code of Hammurabi is not the earliest law code to have been discovered in the region, it is simply the most famous.  The Code of Hammurabi has nothing to do with Mosaic Law.  If you believe that Mosiac Law has something to do with the Code of Hammurabi, despite all evidence to the contrary, go muck up that article.
2. There are other law codes that have been found that were independently created of the Code of Hammurabi, such as the laws that appear in the torah, or those in China.
3. Additionally, no one like Moses ever lived in Egypt, nor did Hewbrew slaves.  A story similar to that of Moses had been floating around the Middle East long before there was a Hebrew people.  Furthermore, it has nothing to do with the argument (that being that the Code of Hammurabi has nothing to do with Mosaic Law.  Bringing it up is a deliberate attempt to obfuscate the topic. 
4. Finally, you have yet to address any of my clearly defined points.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Ahhhh...just woken up!had a bad dream about u tho,so i dont like u right now :) i didnt know anything about comedy night but i guess im up for it.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "
Your point of view is valid and my rv should have been a more detailed ""rv deletion of text without prior discussion or concensus""...or something to that effect. As for it being vandalism...the user has been given many warnings and been rv'd by many users(rv's with summaries, popups, VP etc) as his edits have been considered vandalism by quite a few others besides myself. Ignoring all warnings, attempts to dialogue or simple explanations as to why he refuses to got through proper steps(talk pages) before erasing entire sections, the user still continues his single purpose attack on varying articles. I am certainly not a fan of trivia sections either. But they exist throughout Wikipedia and, like any long standing section in any article, needs discussion and concensus before being turfed. Right? Cheers  "
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "
Talk where you should talk. Not here! You did not wanted to start conversation where it should be started. Do not write here any more. Oui? "
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "

This is a perennially tiresome request that has received stern responseshere and [].  In short; ""not going to happen"".   "
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=violence_or_threats): 13
```
Text: "
American Heritage Dictionary
   1. Physical force exerted for the purpose of violating, damaging, or abusing: crimes of violence.

Online Etymology Dictionary
c.1290, ""physical force used to inflict injury or damage,"" from Anglo-Fr. and O.Fr. violence, from L. violentia ""vehemence, impetuosity,"" from violentus ""vehement, forcible,"" probably related to violare (see violate).

WordNet
1. TABan act of aggression (as one against a person who resists); ""he may accomplish by craft in the long run what he cannot do by force and violence in the short one"" 

Carol Moore User:Carolmooredc User talk:Carolmooredc

Real Restructuring of Article to Explicate POVs

Re-reading the article itself, looking through pages of Search results on violence,  ’s comments on perceptions,  and looking at how the Terrorism article dealt with similar POV problems, I realize the whole article needs restructuring per the below and will do so in next few days after comments.  Honest dealing with POVs is all 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Go chase after her and run her over while she's crossing the street
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Please protect this page. It's not like you get any messages from ips that are not threats of physical torture anyways. Cheers! 

117.201.43.10
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I am getting threats from your sales executive Shifad as i raised complaint against him. Its an official message.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "
I am evil, hear me roar! And don't be fresh! [/me bites Fainites on the ankle. ]   BITE . "
Confidence Score: 0.900
--------------------------------------------------------------------------------
```
