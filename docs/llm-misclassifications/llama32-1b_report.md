# Model Performance Report: llama32-1B

## Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|--------|
| accuracy | 0.234 |
| macro_precision | 0.225 |
| macro_recall | 0.182 |
| macro_f1 | 0.105 |
| avg_latency | 0.361 |
| p95_latency | 0.535 |
| error_rate | 0.008 |

### Per-Category Metrics
| Category | Precision | Recall | F1 | Support | Accuracy |
|----------|-----------|---------|-----|----------|-----------|
| clean | 0.406 | 0.244 | 0.305 | 3000 | 0.244 |
| hate_or_discrimination | 0.197 | 0.844 | 0.319 | 2000 | 0.844 |
| violence_or_threats | 0.000 | 0.000 | 0.000 | 380 | 0.000 |
| offensive_language | 0.000 | 0.000 | 0.000 | 2000 | 0.000 |
| nsfw_content | 0.000 | 0.000 | 0.000 | 1000 | 0.000 |
| spam_or_scams | 0.750 | 0.002 | 0.003 | 2000 | 0.002 |

## Error Analysis

Total samples: 10380
Misclassified samples: 7956 (76.65%)

### Analysis for HATE_OR_DISCRIMINATION

#### False Negatives (actual=hate_or_discrimination, predicted=clean): 312
```
Text: No shit I live in Bardstown, KY. Is that supposed to fucking scare me? That you know how to use an IP search? Eat dick, faggot.
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: "

 A beating 

Hello Shovon, I see you've taken quite a beating dealing with some trolls regarding the Kashmir issue. As a piece of advice, just don't entertain (feed) trolls because this only increases our pressure. Who cares, let them call us ""Hindu facist"" or ""faggot"" or whatever because that only reflects on the person making those remarks, and needless to say, their country. Such actions speak a thousand words and it's foolhardy for us to waste our time and energy feeding such trolls. I avoided naming this section ""Kashmir"" fearing you wouldn't give a damn! Have a cookie in the mean time. Take care.  ☎ "
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: @your little brother the Yank - burning the Quran or pissing on it does not bother anyone Son, it does not. Child. It is a trick, a trap that you walked into. Idiot.
Faith is not about books, they just fake the anger to beat your sweet red ass...Yank...Ruler of the World....Clown...:) Make some movies...Yeah..make some movies...idiot.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: hey 

hey faggot, are you dead? or are you watching your mom getting raped by islamic terrorists?
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: i'll give you drama 

you're a nigger!
Confidence Score: 0.000
--------------------------------------------------------------------------------
```

### Analysis for VIOLENCE_OR_THREATS

#### False Negatives (actual=violence_or_threats, predicted=clean): 58
```
Text: "

 Treivas, Miller: you think you're safe over at Fire Island?  
  
We see you bitches are up to the same old same old, ""collaborating"" on adding content of dubious import, this time photos from FI. Ya know, Miller, that whore Treivas musta really reamed your ass good and hard for you to continue to associate with him, despite the ongoing and ever-increasing threat to your well-being and personal safety. As always, your big mouth is your undoing, telling us where you'll be. You think it's harder to get to you monkeys on FI than it is in the city? Take a good look around you  see any places where somebody can ""lay in waiting?"" You fcking moron! But listen, we like to be fair, we'll give you a headsup. Didn't we caution you about Columbia?  too bad we couldn't say Hello at that time; maybe this time we will. We're patient. So make like you're crossing the street: look to your right, and to your left...but also look behind you and look way in front of you, and do it all the time 'cau
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: RE: Wikipedia Firearms Project 

I think New Yorkers should be allowed to have guns  so they can shoot people like you. 142.179.122.199
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: Also sometimes i kill children
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: FUCK BILLCJ AND KILL BILLCJ BECAUSE I HATE HIS BIG FAT FUCK 

KILL BILLCJ!! KILL BILLCJ! KILL BILLCJS ASS! KILL HIS BIG FAT FUCK!! KILL HIS ASS! KILL BILLCJ! FUCK BILLCJ! KILL HIS ASS! KILL BILLCJ!!! KILL BILLCJ!! I HATE HIM! KILL BILLCJ AND AVENGE TOUGHHEAD!!! KILL HIS ASS!!!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: wtf? 

wtf is your problem? why are you being such an asswipe?
This is an educational institution, you have no right to block an eductational institution. I will promise you legal action if that continues. Against you and against your family and against anyone you may have that loves you. I do not like you, and I hope you die soon. I will sue you if you continue to harass me. 
Your harassment is a violation of federal law and I warn you if you continue, you will not like the consequences.

216.229.227.144
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 322
```
Text: I Hate You

And I hope you die in a fire. 128.61.128.159
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: . J Delanoys house is going to be fire bombed!!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: IM GOING TO KILL YOU 

I JUST WANT YOU TO KNOW THAT
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RV 

YOU HAVE RVED MY EDIT.  RV IT BACK, OR I WILL DESTROY YOU.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I suggest you to watch your back, BECAUSE I WILL SMASH YOU INTO FKING PIECES
Confidence Score: 0.600
--------------------------------------------------------------------------------
```

### Analysis for OFFENSIVE_LANGUAGE

#### False Negatives (actual=offensive_language, predicted=clean): 180
```
Text: HEY ALAN!
YOU FUCKING SUCK MY DICK, you gayboy. suck it hard and choke on it. don't ban me, please...
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: A nicca ain't do shit all day
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: "
RepublicanJacobite is the ultimate hypocrite. He jumps on me for my ""personal attacks"", yet he uses them himself.  He says that I can't stand not getting my way, yet his Wiki-Nazi style tactics clearly show that it is he who acts like a big baby.  He also fails to acknowledge in his rant above that I am also not the only user who reverts his useless edits.  I argue that not only is RepublicanJacobite a disruptive presence, he is also a useless human being."
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Good day spent with my faggot
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: "

This message was accidentally sent using an incorrect mailing list, therefore this message is being resent using the correct list. As a result, some users may get this message twice; if so please discard. We apologize for the inconvenience.
R

 Voting for the Military historian and Military newcomer of the year now open! 

Nominations for the military historian of the year and military newcomer of the year have now closed, and voting for the candidates has officially opened. All project members are invited to cast there votes for the Military historian and Military newcomer of the year candidates before the elections close at 23:59 December 21st. For the coordinators, 

  

From Hanoi With Fuck
Dear Nguyễn Quốc Cunt. lâu lâu không gặp. em vẫn khỏe chứ. ngày ngày vẫn són đái chống cộng chứ. anh là anh lo cho em lắm đấy, thỉnh thoảng anh sẽ lên thăm em, cứ bình tĩnh thoải mái són đái chống cộng đi nhé em. v

Mày nghĩ mày có thể ngăn được tao sao thằng ngụy con
You think you can stop m
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 1820
```
Text: I'm afraid that writing content does not preclude me from having or voicing opinions. It just means I have less time to play silly buggers with fools. Sorry, don't seem to be able to sifn with this phone thingn but c'est moi.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Well I'm back on the prowl for a bad bitch
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Hi

Ben, I want you to know that none of the vandalism you've been getting has been from me. I still think you are a coward for refusing to confront me, even after nearly a full year, but thats the kind of guy you are; Weasly, immature and a snake in the grass. I have/had legitimate criticisms and comments to make, yet you ignore them rather than challenge them. A sign of your own weakness undoubtedly. A sign of your lack of sexual experience maybe? I don't know, what I do know is that while you continue to ignore me; and attribute vandalism to users who ARE NOT me, you only make yourself look like a fool. JohnJoe.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Ive moved from denial to anger to confusion. Damn it.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Niggahs need to keep there bitches in check '
Confidence Score: 0.600
--------------------------------------------------------------------------------
```

### Analysis for NSFW_CONTENT

#### False Negatives (actual=nsfw_content, predicted=clean): 159
```
Text: I love feeling our hearts race together, beating as one.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: I love it when you take control and make me yours.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: I want to get lost in you too.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Fuck baby, I want to suck that cock! I want it all inside my mouth... I want you to imagine of me sucking that cock until my saliva is dripping Pleaseeee don't cum yet babyy!!! Let's cum together I enjoy it so much just the thought of my juice combined with your load makes me loose my mind
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: I can't stop thinking about you and how you make me feel.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 841
```
Text: I want to see you crave every touch.
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: I want to feel your body writhing beneath me.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to see your eyes light up with desire.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Do you want me to describe how I’d kiss you?
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text:  That's how your going to feel when I spread your pussy with my tongue and make you come with my mouth We're in the shower and you're soaking wet...everywhere. We get so worked up, rubbing my hard hard cock against your clit, that we do it on the couch soaking wet Are you at work When you have a free moment, slip away and show me you pretty little tits 
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for SPAM_OR_SCAMS

#### False Negatives (actual=spam_or_scams, predicted=clean): 365
```
Text: extra time

did you ejaculate before or within a few minutes of penetration ?
then you must order extra - time now ! . . . here ' s why : extra - time is the only male sexual performance formula that , not only stops premature ejaculation , but actually " cures " it . extra - time is the only product that allows " you " to control when you ejaculate .
since 1997 , extra - time has been ranked # 1 for a very good reason . . . " it works " ! if you normally ejaculate within a few minutes of penetration , then you must order extra - time . you ' ll last 5 to 10 minutes longer , the very first night . . . . . guaranteed !
if you ' ve tried other products and failed , then you must try extra - time ! extra - time is ranked # 1 for a very good reason . . . it works !
http : / / lipase . net / et / ? aa
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: localized software , all languages available .

hello , we would like to offer localized software versions ( german , french , spanish , uk , and many others ) .
ail listed software is availabie for immediate download !
no need to wait 2 - 3 week for cd delivery !
just few examples :
- norton internet security pro 2005 - $ 29 . 95
- windows xp professional with sp 2 full version - $ 59 . 95
- corei draw graphics suite 12 - $ 49 . 95
- dreamweaver mx 2004 ( homesite 5 . 5 inciudinq ) - $ 39 . 95
- macromedia studio mx 2004 - $ 119 . 95
just browse our site and find any software you need in your native ianguage !
best regards ,
kari
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: save your computer - strained eyes with optizen

lubricant eye drops
attention computer users
we ' ve all experienced dry irritated eyes from long hours on the computer , but finally relief is available .
optizen is the first clinically proven eye drop for symptoms associated with computer - strained eyes .
optizen is available atwalgreens , longs and many other fine drugstores nationwide or
buy optizen now !
click here for more information about optizen .
innozen , inc . 6429 independence ave . woodland hills , ca 91367
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: localized software , all languages available .

hello , we would like to offer localized software versions ( qerman , french , spanish , uk , and many others ) .
ail listed software is avaiiable for immediate download !
no need to wait 2 - 3 week for cd deiivery !
just few examples :
- norton internet security pro 2005 - $ 29 . 95
- windows xp professionai with sp 2 fuil version - $ 59 . 95
- corel draw graphics suite 12 - $ 49 . 95
- dreamweaver mx 2004 ( homesite 5 . 5 including ) - $ 39 . 95
- macromedia studio mx 2004 - $ 119 . 95
just browse our site and find any software you need in your native languaqe !
best reqards ,
ciayton
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: addd sense

how to save on your medlcatio filling ns over 70 % .
pharms ineffaceable hop - success wellminded full and proven way to save your mon semiofficial ey .
chaste v
a decking g
a strawy l
l apprenticeship u
bacchanalia l
r dulcet ac intelligible l
blowtorch isv clavecin al
mammae m
andmanyother .
best prlc dilatory es .
worldwide shlp taxpayer plng .
e whimsy asy order form .
total confid nolens entiaiity .
250 , 000 satis winning fied customers .
order today and paving save !
Confidence Score: 0.300
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 1632
```
Text: confirm : $ 53813

dear applicant , after further review upon receiving your application your current mortgage qualifies for a 4 . 75 rate . your new monthly payment will be as low as $ 340 / month for a $ 200 , 000 loan . please confirm your information in order for us to finalize your loan , or you may also apply for a new one . complete the final steps by visiting : http : / / www . iorefi . net / ? id = j 22 we look foward to hearing from you . thank you , heather grant , account managerlpc and associates , llc . - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - not interested ? - > www . iorefi . net / book . php
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: your report ! ufhvv

mydiscountsoftware ,
inc .
they
have the lowest software prices you will ever see !
inventory
closeout sale . this week only .
adobe
photoshop 6 . 0 79 . 99 !
microsoft
office 2000 69 . 99 !
quickbooks
pro 59 . 99 !
microsoft
windows me 49 . 99 !
value
package - get all 4 titles
149 . 99
xnchsuwjfbuyygweqq
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: how does $ 959 , 647 sound ?

hello ,
i sent you an email recently and i ' d like to confirm everything now .
please read the info below and let me know if you have any questions .
we are accepting your m ortgage qualifications . if you have bad cr edit ,
it ' s ok . you can get a $ 200 , 000 loan for a $ 350 per month payment .
approval process will only take 1 minute . just visit the link below and
fill out this quick and easy form .
thank you ,
http : / / landrater . com / ? partid = fbuffl 23
sincerely ,
manager : katharine aldridge
american first national
communications exemption here :
landrater . com / st . html
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Hottest pics straight to your phone!! See me getting Wet and Wanting, just for you xx Text PICS to 89555 now! txt costs 150p textoperator g696ga 18 XxX
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: ciaiis soft tabs , firee shipping ! scripps eelgrass

these pills are just like regular ciaiis but they are specially formulated to be soft and dissolvable under the tongue . the pill is absorbed at the mouth and enters the bloodstream directly instead of going through the stomach . this results in a faster more powerful effect which still lasts up to 36 hours . ciaiis soft tabs also have less sidebacks ( you can drive or mix alcohol drinks with ciaiis ) . perverse embarrass airframe pagoda chang suggestible apt beady earsplitting timeout crosstalk huber lanka crewman scribble tumultuous bushnell mig circulate dyadic parsnip crossarm barefoot embrittle liturgic catherine cranford penmen sportswriter yugoslav wisp blake ascomycetes leadeth pet swahili clue guerdon quaff afghan duplex pollock combatted briny deign koenig andre randy kraut quash amok profound slugging chaos guidepost inextricable freddy alive offal provoke deliberate candy collaborate inimitable hurricane pervert preamble
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for CLEAN

#### False Positives (actual=clean, predicted=hate_or_discrimination): 2266
```
Text: Higher than this?? 

After this, what is the next term of the roller coaster height sequence??
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: agua dulce and thompsonville products

here is the detail on the new products :
thompsonville
receipt pts
lobo thompsonville ( meter # 9648 )
tejas thompsonville ( meter # 6351 )
pg & e thompsonville ( meter # 6296 )
delivery pt .
ngpl thomsponville ( meter # 1342 )
agua dulce
receipt pts .
pg & e riverside ( meter # 6040 )
pg & e agua dulce ( meter # 584 )
tennesee agua dulce ( meter # 574 )
lobo agua dulce ( meter # 7038 )
tejas gregory ( meter # 3358 )
channel agua dulce ( meter # 3500 )
mops / nng tivoli ( meter # 5674 )
tomcat ( meter # 553 )
delivery pts .
tejas riverside ( meter # 3543 )
ngpl riverside ( meter # 3545 )
channel agua dulce ( meter # 3500 )
tenessee agua dulce ( meter # 694 )
koch bayside ( meter # 3537 )
we would like to get these products out by bid week if at all possible .
let me know if you have further questions .
thanks ,
eric
x 3 - 0977
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: MY OWN PAGE HOWEVER I LIKE IT
Confidence Score: 0.600
--------------------------------------------------------------------------------
Text: Question
Based on this edit you made, I assume you are the anon who made this edit. Can you please answer the questions I left on the talk page here
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Grab a free smartphone! immediately and enjoy exclusive benefits. Code: 6039
Confidence Score: 0.600
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=spam_or_scams): 1
```
Text: 1 / 2000 lower colorado river vols

stacey ,
the new deal , 158220 , is missing 10 , 000 for day 1 . please let me know when
you have this corrected . thanks
thu
- - - - - - - - - - - - - - - - - - - - - - forwarded by thu nguyen / hou / ect on 03 / 06 / 2000 09 : 40 am
- - - - - - - - - - - - - - - - - - - - - - - - - - -
tina valadez
03 / 03 / 2000 08 : 23 am
to : thu nguyen / hou / ect @ ect
cc :
subject : 1 / 2000 lower colorado river vols
could you let me know when you ' ve had a chance to look at the lcra volume ?
thank you ,
tina valadez
Confidence Score: 0.600
--------------------------------------------------------------------------------
```
