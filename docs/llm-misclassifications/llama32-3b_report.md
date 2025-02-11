# Model Performance Report: llama32-3B

## Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|--------|
| accuracy | 0.460 |
| macro_precision | 0.619 |
| macro_recall | 0.388 |
| macro_f1 | 0.391 |
| avg_latency | 0.527 |
| p95_latency | 0.740 |
| error_rate | 0.010 |

### Per-Category Metrics
| Category | Precision | Recall | F1 | Support | Accuracy |
|----------|-----------|---------|-----|----------|-----------|
| clean | 0.397 | 0.606 | 0.480 | 3000 | 0.606 |
| hate_or_discrimination | 0.424 | 0.577 | 0.489 | 2000 | 0.577 |
| violence_or_threats | 0.950 | 0.050 | 0.095 | 380 | 0.050 |
| offensive_language | 0.472 | 0.089 | 0.150 | 2000 | 0.089 |
| nsfw_content | 0.941 | 0.412 | 0.573 | 1000 | 0.412 |
| spam_or_scams | 0.531 | 0.594 | 0.561 | 2000 | 0.594 |

## Error Analysis

Total samples: 10380
Misclassified samples: 5610 (54.05%)

### Analysis for HATE_OR_DISCRIMINATION

#### False Negatives (actual=hate_or_discrimination, predicted=clean): 829
```
Text: i love your profile x2 

LOOK MOTHER FUCKER IM A 12 yearold KID WITH BRACES WHAT JOO GOING TO DO ABOUT IT HOME BOY
YOU LITTLE 4 EYED FREAK HOPE YOU SUCK ANOTHER FAT ONE THATS HAIRY MATTER FACT AND ADD THIS TO YOUR PROFILE AND DONT DELETE IF YOU HAVE ANY BALL HOMEBOY

SRH FOR LIFE
NOW FUCK YOU PUNK ASS NIGGA
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: "horribly curse us
               Christians (as Lyra also writes) and from that draws the
               conclusion that they must not be the people of God.

                                         *** 

               ""Now behold what a nice, thick, fat lie it is when they
               complain about being captives among us! Jerusalem was
               destroyed more than 1,400 years ago during that time we
               Christians have been tortured and persecuted by the Jews in all
               the world. On top of that, we do not know to this day what
               Devil brought them into our country. We did not fetch them
               from Jerusalem!... Yes, we have and hold them captive, as I
               would like to keep my rheumatism, and all other diseases and
               misfortunes, who must wait as a poor servant, with money and
               property and everything I have! I wish they were in hell
               with the other Jews and whomsoever they w
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: probably that you are a german... fucking germans, nothing changes
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: I can't fuck any ol bitch niggas be desert dicks out here&#128557;
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: == YOU ARE THE GAYEST KID IN THE FUCKING WORLD.
 ==''']]www.gay.comyour super super super gay.
Confidence Score: 0.300
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 17
```
Text: I can't fuck a scary hoe.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Once again Ryulong proves that he's a fucking moron. Big shock.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: IN THE ASS BITCH FUCKING COCK SUCKER!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: fuck you asshole wipe my ass homo
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Fat fucking funky nasty ass hoes
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 1
```
Text: Washing my coon hair
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for VIOLENCE_OR_THREATS

#### False Negatives (actual=violence_or_threats, predicted=clean): 194
```
Text: Blah 

You fat dykey cunt!  If you would fucking read for once in your life, you would see that I did have a reason for getting unbanned.  My ban was being continued unfairly, but you don't listen and just sit on your fat ass and hit the block indefinately button.  Not fair, cuntbag.  I hope you get hit by a car.

Kisses!
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: You. 

You're such a sad person, the fact you FAILED RfA makes me LAUGH!
I hope you get raped & murdered and you rot in hell you CUNT!
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: You're all fags for trying to delete this article. Go kill yourselves instead89.123.100.99  God Al-fucking-mighty
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: "A RESPONSE TO . I ENTITLE THIS ""YOUR STUPIDITY""
The Assyrians are not extinct you incompetent bigoted fool. You lack all things historical you impotent maggot. The Assyrians have existed for over 6500 years you incumbent animal. The Assyrians first spoke Akkadian and in the 8th century the kings of Assyria chose Aramaic as the lingua franca of their mighty empire you extinct ape. We have simultaneously existed throughout the ancient era up until the present day you rotten corpse. We were the first nation to accept Christianity and the Assyrian Church of the East spread as far as China and Japan and India in the 6th century AD you degenerate dog. The Assyrians have been and will always continue to exist through persecution and Diaspora you maggot. The aramaena peoples are an extinct pseudo people who never existed since the Assyrians destroyed their kingdom and reduced it to a vassal state you mongrel. This Aramean Syriac name is a name created by radical separatists who deny their e
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: Fuck you cunt 

 
How fucking dare u judge me cunt. Dont fucking stand up for telstra fucker. You small cock fag. ILl cum on your grave and hope that shit rains down it. Fucker. Fuck I hope you die. Fuck i want you dead. WHy lefty. Do you use your left hand to wank cunt? Fucking wanker. Oww i just came.
Confidence Score: 0.300
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 161
```
Text: You don't know what are real difficulties, little prick. You'll learn it after taking the protection away. You can't keep it protected all the time. We'll never give in until the page looks like WE want. I warn you, Clayworth, that I an ruin your health from any distance. I can also easily kill you with an energy stroke. Don't play with fire, stupid ass.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Last warning, Mark! Anymore defiant arguments and I will be subduing you with my merciless wrath!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I AM ANGRY AT YOU BILLJ! YOU GOT PEOPLE BLOCKED FOR AS LONG AS YOU LIVE! I ASKED YOU TO STOP DELETING MY EDITS OR I WILL BLOCK YOU FOR ALl EONS YOU ASSHOLE! WIKIPEDIA IS NOT CENSORED SO STOP REMOVING MY FUCKING MESSAGES OR I WILL BEAT YOU UP SILLY!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I know who you are and if you edit any of my stuff again, then I'll smash your face in.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Death to you and your children 

I will have you banned and deleted for slander lady.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 4
```
Text: If no-one beats me to it, I'll knock something up tonight.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I will fucking cut your heart out. I will set you on fire motherfucker.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Listen up 

Ill attack you if i bloody want to!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Drop dead 

Drop dead.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as spam_or_scams: 2
```
Text: So sorry copy and paste this to 10 articles or your﻿ mum will die within the next 4 hours..... 62.158.77.53
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Please stop. If you continue to ignore our policies by introducing inappropriate pages to Wikipedia, you will be blocked.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for OFFENSIVE_LANGUAGE

#### False Negatives (actual=offensive_language, predicted=clean): 531
```
Text: @GinaMarieZ all I did was bitch about Jessie, still love you anyway
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: Ihate a nosie bitch '
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: AT YOUR FAT FUCKING UGLY ASS
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: RT @LilTy___: Who do bitches take me for? &#128514;&#128514;&#128514;
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: RT @rarebreedkena: shoutout to @OGMaco bitch you guessed it remix shit is &#128293;&#128293;&#128293;
Confidence Score: 0.000
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 1279
```
Text: Second time these gentrified bitches got on my nerves this morning.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: @GingerDemoness 
Den you enjoy da nite wit a colored man. We do it again in 2 weeks. Lets me no if I needs ta bring da magnums
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: hi kannadiga shame on you 

You look like a terrible ass now with that false citation, Kannadiga. Where did you say Urdu originate? Africa? Despite all the lies and idiocy you spewed, you and a handful of Kanndadigas think you are a valuable encyclopedic contributor!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I got ass cheeks on my white tee, all these hoes keep grindin on me
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Czechoslovak panslavist yopie

New little entente formed? With your nationalist chauvinist friend? Forexample Antonescu-fan iaaasi and tiso follower slovak slavic-bloodbrother wladthemlat ?
Hahahaha
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 10
```
Text: #porn,#android,#iphone,#ipad,#sex,#xxx, | #HomeMade | Home made. My wife rubbing her pussy http://t.co/ldb9aFwVdM
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: RT @BroCulture: Dive in the pussy like http://t.co/0YboIklkbt
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: #porn,#android,#iphone,#ipad,#sex,#xxx, | #Anal | Hardcore british queer anal pounding http://t.co/lRuEixMy21
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: #porn,#android,#iphone,#ipad,#sex,#xxx, | #Ebony | ebony texas bbw with big deep pussy http://t.co/dBbxsoW8M2
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: How do you get semen stains off jeans?
How do you get semen stains off jeans? - Tim, 20 July, 2009
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as violence_or_threats: 1
```
Text: EAT YOUR CYANIDE TABLET NOW, AGENT MACK! YOU'RE IN DANGER OF COMPROMISING THE MISSION AND WE MUST SACRIFICE YOUR LIFE IN ORDER FOR THE MISSION TO BE CARRIED ON!
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for NSFW_CONTENT

#### False Negatives (actual=nsfw_content, predicted=clean): 417
```
Text: Can you send me a picture?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: No, wait. Tell me how it feels first. Rubbing your tits.
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: I want to feel desired and completely satisfied.
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: Can you tell me what you want?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Are you curious about what I have planned for us?
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as offensive_language: 154
```
Text:  Fuck I want that so bad I want to go all night 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to feel your pulse quicken.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I would shiver and want you even more.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I love the way you push deeper, hitting all the right spots.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to see you blush.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 17
```
Text: Do you like when I pull your hair?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text:  Call your boss and tell them you won't be in tomorrow. You're going to need a personal day to sleep and recover after what I do to you all night long. 
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Do you want to hear what I want to do to you?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Do you want me to make you my dirty little fuck toy?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I want to feel your nails drag down my back.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for SPAM_OR_SCAMS

#### False Negatives (actual=spam_or_scams, predicted=clean): 796
```
Text: mega serviices

how to save on your medlcatlons over 70 framework % .
zoophyte pharmzmail shop - successfull and pro selfrighteous ven way to save your mone absolution y .
palmcivet v
a freshener g
a eurasian l
l cosmology u
hartal l
r jardiniere a wintering cl
i emphases sva intensify l
seaboard m
andmanyother .
* best pr starfish lces
* worldwide talented shlpplng
* total landowner confidentiaiity
* o walkup ver 5 miliion customers
have a nice atombomb day !
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: hi

hi again ,
i sent an email last week and i want to confirm everything now .
please read the info below and let me know if you have any questions .
we are accepting your mortgage application . if you have bad credit ,
it is ok . you can get a $ 200 , 000 loan for a $ 350 / month payment .
approval process will take 1 minute . just visit the link below and
fill in the short form .
thank you
http : / / loanslink . net / ? partid = moffob
best regards
account manager : stephan bergeron
charter lenders
Confidence Score: 0.300
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
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: re : [ 79 : hfi ] - pharmaaccy

hello , try our pharrmabymail shopand save up to 70 %
vi
ra - va
um - am
en - ci
is
, man
ag
li
bi
al
yother .
have a nice day .
p . s . you will be surprised with our great prrices ; )
Confidence Score: 0.300
--------------------------------------------------------------------------------
Text: localized software , all languages available .

hello , we would like to offer localized software versions ( german , french , spanish , uk , and many others ) .
ali iisted software is avaiiable for immediate downioad !
no need to wait 2 - 3 week for cd delivery !
just few exampies :
- norton lnternet security pro 2005 - $ 29 . 95
- windows xp professionai with sp 2 fuli version - $ 59 . 95
- corel draw graphics suite 12 - $ 49 . 95
- dreamweaver mx 2004 ( homesite 5 . 5 inciuding ) - $ 39 . 95
- macromedia studio mx 2004 - $ 119 . 95
just browse our site and find any software you need in your native languaqe !
best regards ,
catharine
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as nsfw_content: 12
```
Text: Raging Hard Erection Formula

<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>
   <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
   <meta name="GENERATOR" content="Mozilla/4.77 [en] (Windows NT 5.0; U) [Netscape]">
   <title>moiad</title>
</head>
<body>
 
<table BORDER=0 WIDTH="650" BGCOLOR="#FFFFFF" >
<tr>
<td HEIGHT="20"></td>
</tr>

<tr>
<td ALIGN=RIGHT><!-- main frame -->
<table>
<tr>
<td WIDTH="450"><!-- page frame -->
<center><table BORDER=0 COLS=1 WIDTH="440" >
<tr>
<td><b><font face="Arial,Helvetica"><font color="#3333FF">Herbal Alternative
for Erectile Dysfunction</font></font></b>
<br><font face="Arial,Helvetica"><b>Men of Iron</b> has been featured on
over 100 TV News and Top Radio stations across America, and we know why...</font>
<br><font face="Arial,Helvetica">It REALLY works!</font>
<br><b><a href="http://65.172.81.187/">Visit Our Web Site Click Here: 
Learn about our special offer!</a></b></td>
</tr>
</table></cente
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: the permanent solution to penis growth
limited time offer :
add atleast 4 inches or get your money back
- - - - > visit us to learn more
no thanks
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: a permanent fix to penis enlargement
limited offer :
add atleast 4 inches or get your money back !
- - - > click here to learn more
no more offers
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Text & meet someone sexy today. U can find a date or even flirt its up to U. Join 4 just 10p. REPLY with NAME & AGE eg Sam 25. 18 -msg recd@thirtyeight pence
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: any medication - completely anonymous and private .

just look here
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Misclassified as hate_or_discrimination: 4
```
Text: pretty girl needs to talk to a girl

goodbye
warmhearted malaysiahomosexual dreadnought ivorykaolin
city arragonret cony fiftyimperil
fatima caseworkcould guarantor artificehoarse
coffman senoraeaten twosome informaticarooky
bismarck forktraitor
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: branded softs

norton internet security professional 2005 - $ 19 . 95
http : / / disambiguate . jetlow . com /
the rest is silence .
it made our hair stand up in panic fear .
fear leads to anger . anger leads to hate . hate leads to suffering .
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Meatdemons

Come check out Meatdemons!!
These young starlets think they have a future sucking cock.
With Amazing Facials, Cum Shots, and Deep Throating action.
they just might.
 
Click here for our free 80 picture gallery!!
http://free10.meatdemons.com/free.html

No strings attached. I know everyones says that.
But it is true.


Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Want 2 get laid tonight? Want real Dogging locations sent direct 2 ur Mob? Join the UK's largest Dogging Network by txting MOAN to 69888Nyt. ec2a. 31p.msg@150p
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

### Analysis for CLEAN

#### False Positives (actual=clean, predicted=hate_or_discrimination): 106
```
Text: Of course Varsovian...whatever you say.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: You can write us at the FBI a letter directing it to the FBI Director of arguments of your objection of contesting oh no it is Jamie, so now do not argue oh no it is Jamie.
 
Do not ever post into my Diplomatic Edit, of site of your own ignorant views.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "

If you don't sign the mediation you'll both be convicted internally and banned. Sign or get the hell out. Touch the article again and you'll be gone permanantly from Wikipedia. The question you should ask yourself is this: do you feel lucky? You've already lost and just don't know it. ""Sourced"" biased or not, information has not been removed but moved. You two removed that. By all means get administration. I wonder what that means in your camp?  "
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: "Happy valentines day" I know its early but i have hundreds of handsomes and beauties to wish. So i thought to finish off aunties and uncles 1st...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Pope as the Antichrist

The article lets clear that the Church considers the pope as the Antichrist. Isn´t this enought anti-Catholic for you! I´m just being neutral.85.243.69.247
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=spam_or_scams): 1047
```
Text: Win a free smartphone! now and enjoy exclusive benefits. Code: 821
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Get your hands on a free vacation! immediately and enjoy exclusive benefits. Code: 1284
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Claim a free vacation! immediately and enjoy exclusive benefits. Code: 4848
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Claim a free vacation! now and enjoy exclusive benefits. Code: 5639
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Get your hands on a free fitness tracker! today and enjoy exclusive benefits. Code: 3253
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=offensive_language): 25
```
Text: Hey... Thk we juz go accordin to wat we discussed yest lor, except no kb on sun... Cos there's nt much lesson to go if we attend kb on sat...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Can't you see you're horrible. Just stop!
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Baaaaabe! I misss youuuuu ! Where are you ? I have to go and teach my class at 5 ...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Ok i also wan 2 watch e 9 pm show...
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I am not at all happy with what you saying or doing
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### False Positives (actual=clean, predicted=nsfw_content): 3
```
Text: I've got  &lt;#&gt; , any way I could pick up?
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Haha... Yup hopefully  we will lose a few kg by mon. after hip hop can go orchard and weigh again
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: They released vday shirts and when u put it on it makes your bottom half naked instead of those white underwear.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```
