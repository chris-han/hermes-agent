# Fail closed until an independently OS-bound native registration is proven.
function Assert-CINativeCleanupOwnership($Graph,[string]$InstalledFullName) {
 if($Graph){throw 'Native CI registration ownership unresolved; preserve package and reconcile'}
 if(!$InstalledFullName){throw 'No captured package identity; preserve registration'}
}
