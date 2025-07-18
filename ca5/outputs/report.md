# Lyrics Clustering Report

**Dataset size:** 2,999 songs

## Pre‑processing Comparison
* Stemming pipeline — best K‑Means silhouette = 0.056 at K = 2
* Lemmatisation pipeline — best K‑Means silhouette = 0.063 at K = 2

We proceeded with **lemmatisation** because it yielded the higher silhouette score.

## Final Clustering Scores
| Algorithm | #Clusters | Silhouette | Notes |
|-----------|-----------|------------|-------|
| K‑Means | 2 | 0.063 | elbow + silhouette selected |
| DBSCAN | 1 | 0.154 | eps = 0.50 |
| Agglomerative | 3 | 0.036 | Ward linkage |

## Cluster Examples (K‑Means)
### Cluster 0
* Cryptic psalms Amidst the howling winds A scorching source of agonizing bliss Beneath its veil Mysteries of a life beyond Can you hear it? Sons and daughters with hearts ablaze Forsaken ones in deaths embrace Chant this hymn, behold in awe The blessed curse, abort by law Come reign with us Enslave yourself unto the beast Extend your lust Tyrants of dark supremacy Sons and daughters with hearts abl…
* Im sleeping tonight with all the wolves Were dreaming of life thats better planned As long as the wind that falls isnt longing for revenge I should be safe We should be safe Shes two bitter ends So watch as those friends Enjoy suns embrace when Stories theyve told Through ears on the walls Speak softly to them all…
* Wings of the darkest descent Fall from the realm of dark From the blackest fall of creation Doomed by its end Winds of chaos blow through my soul Wings of the darkest descent shall fall Lurking evil shadowed darkness Far beyond the light of day Blessed be the wings of chaos Blowing through your soul When evil consumes my creation Wings of the darkest descent shall fall Black descends on the wings…

### Cluster 1
* [Verse 1] Norrid Radd was my real name Had a job that I hated every day Until that one day I told my boss To just shove that damn JOB Cause I found a place I could rest my head, maybe call my home Trapped on this planet I know Found a bunch of other super powered people like before Fought against evil on my silver surfboard Cause everybody wants the power cosmic Skate across space thats our depart…
* Yeah [Verse 1 Classified] They might say I lead a simple life, the type that dont excite And yes I like to smoke more than light til night I realize I got a problem, Im high way too often And only seem to acknowledge it after I indulge in it Pull in my driveway, walk through my front door Then make a bowl of cereal to conquer my hunger See I like to eat before I slumber Brushin my teeth and gettin…
* Dont look at me that way Like you know how its going to end.. Cause am........too tired to be honest [Verse 1 Th3 Ghost] I hope me and money be in holy matrimony And on our moneymoon We will have fun If we have a sonsun He will light up the world If we have a girl She will be mother earth And her worth? Worthless to the earthlings Who dont know the worthy But the worldly They dont want the spiritu…

---
**Homogeneity** was _not_ computed because the dataset does not contain ground‑truth genre labels.
