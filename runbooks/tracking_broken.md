# Tracking broken

## Tracking pixel missing
If the landing page is up but the conversion pixel is missing, a recent deploy most likely removed the tracking snippet. Ask the tracking team to restore the pixel and compare the page source with the previous release.

## Postback not received
Pixel events fire but postbacks do not arrive. Check the advertiser postback URL, the security token and the tracker error log. After the fix, reconcile missed conversions against the advertiser report so affiliates are paid correctly.
