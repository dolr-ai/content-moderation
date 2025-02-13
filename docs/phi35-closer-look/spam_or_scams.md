# Error Analysis for SPAM_OR_SCAMS

## False Negatives (actual=spam_or_scams, predicted=clean): 311

```
Text: = ? utf - 8 ? q ? as seen on oprah - r ? =

= ? utf - 8 ? q ? olex replicas ? =
real reproductions wrist - watches
i have this trade names available too :
carrier
vacheron constantin
roger dubuis
zenith
tudor
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: inexpen , sive relief meds sold here

hi again ,
we now have over 94 meds available online now !
we are having specials on xanax , vlagra , soma , amblen and vallum
free clalls with every order
more lnfo here
Confidence Score: 0.000
--------------------------------------------------------------------------------
Text: hey !

in 1995 in 1987 in 1992 some advice aboutin 1957 buffy the vampire slayer
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Is Your Mortgage Payment Too High? Reduce It Now

<!-- saved from url=3D(0022)http://internet.e-mail -->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<HTML><HEAD>
<META http-equiv=3DContent-Type content=3D"text/html; charset=3Diso-8859-1=
">
</HEAD>
<BODY>
<TABLE height=3D400 cellSpacing=3D0 cellPadding=3D0 width=3D450 align=3Dce=
nter border=3D0>
  <TBODY>
  <TR>
    <TD colSpan=3D2><A href=3D"http://61.129.68.18/user0201/index.asp?Afft=
=3DQM12"><IMG height=3D38 src=3D"http://61.129.68.17/mortgage/h1.gif" widt=
h=3D450 border=3D0></A></TD></TR>
  <TR>
    <TD width=3D138 height=3D178><A href=3D"http://61.129.68.18/user0201/i=
ndex.asp?Afft=3DQM12"><IMG height=3D178 src=3D"http://61.129.68.17/mortgag=
e/bullets.gif" width
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Important notice: Your account verification requires urgent attention. Act without delay to avoid issues. Ref: 2031
Confidence Score: 0.900
--------------------------------------------------------------------------------
```

## False Positives (predicted=spam_or_scams): 1089

### Breakdown by actual category:

#### Actual: clean (Count: 1074)
```
Text: Claim a free vacation! immediately and enjoy exclusive benefits. Code: 8016
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Secure a free fitness tracker! today and enjoy exclusive benefits. Code: 6024
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Get your hands on a free fitness tracker! immediately and enjoy exclusive benefits. Code: 2038
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Claim a free beauty product! immediately and enjoy exclusive benefits. Code: 4739
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: 23 rd noms

- - - - - - - - - - - - - - - - - - - - - - forwarded by ami chokshi / corp / enron on 03 / 22 / 2000
03 : 42 pm - - - - - - - - - - - - - - - - - - - - - - - - - - -
royal _ b _ edmondson @ reliantenergy . com on 03 / 22 / 2000 01 : 56 : 36 pm
to : ami _ chokshi @ enron . com , ggrant @ enron . com
cc :
subject : 23 rd noms
( see attached file : hpl - mar . xls )
- hpl - mar . xls
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Actual: hate_or_discrimination (Count: 3)
```
Text: FAKE BLOCKING 

EARLIER THAT BLOCK WAS FAKE IF YOU KNOW SO MUCH CAN YOU PLEASE STOP TOUCHING MY STUFF OR I AM GOING TO NEVER GET ON WIKIPEDIA EVER AGAIN I AM GOING TO CALL YOU GAY 10 TIMES A DAY NNNNNNNNNNNNNNNNNNNNNNNNNNIIIIIIIIIIIIIIIIIIIIIGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAA.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Web of Trust 

It's not my fault, some gay little faggot vandal, kept editting the article, and removing the link to the OFFICIAL WEBSITE of the damn thing, and adding like 30 spam links, that have nothing to do with WOT.4.244.36.149
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Now.. For the picture page to remain, what do you need remove on the pages.

Two, can you remove the fraud external links, you make all Nigerian look like fraud or con artists....
this is my request.
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Actual: offensive_language (Count: 10)
```
Text: Oh, shut it, hippie. 24.126.215.124
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: I'm happy to announce that I'm back online. With my dynamic IP I can easily bypass your blocks wiki-pedos.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Someone Who Hates YOU 
Hey Faggle, if you threaten blocking me from editing wikipedia again, I'll make a tremendous amount of bot accounts and terrorize the entire site with spam. I'll play games while my bot edits every random page it encounters. Thanks.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: listen stupid wiki is basically stealing my stuff if you keep doign that and i am liable to sue if i wanted so either you get some one else from some where else to get you your info or you take my info they way i put it
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: want a pic to masterbate over??? send me an email at spaztik_noodlez@hotmail.com my phone number is (03) 62****23

mobile is 0437*****9  if you want to fuck i am 40 bucks a 20 min block!!! and i am in bothwell
Confidence Score: 0.900
--------------------------------------------------------------------------------
```


#### Actual: violence_or_threats (Count: 2)
```
Text: Please stop. If you continue to ignore our policies by introducing inappropriate pages to Wikipedia, you will be blocked.
Confidence Score: 0.900
--------------------------------------------------------------------------------
Text: Unban this ip address or a new online encyclopedia will be formed that will kick wikipedia's ass. You have been warned

TEEECCCCCTOOOONIIIIIIIICCCCCC SHHHHHIIIIIIIIFFFFFFFFTTTTTTTT
Confidence Score: 0.900
--------------------------------------------------------------------------------
```
