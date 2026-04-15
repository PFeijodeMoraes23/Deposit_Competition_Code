$files = Get-ChildItem -File -Filter "*.py"
$files += Get-ChildItem -File -Filter "*.sh"
$files += Get-ChildItem -File -Filter "*.txt"

foreach ($file in $files) {
    if ($file.Name -like "d_rate_scrape_*") { continue }
    
    $content = Get-Content $file.FullName -Raw
    $changed = $false

    # Decrease mapping: 6->5, 5->4, 4->3, 3->2
    # We must do it carefully to not cascade. 
    # Use distinct temporary markers if needed, or just replace sequentially if doing it right. Wait, sequential replace 6->5, then 5->4 will cascades 6->5->4.
    # Let's use regex matching and a mapping.

    $content = [regex]::Replace($content, '(rout_|estimation_|export_|submit_blp_hpc_|submit_blp_hpc_test_)([3456])', {
        param($m)
        $prefix = $m.Groups[1].Value
        $num = [int]$m.Groups[2].Value
        $newNum = $num - 1
        return "${prefix}${newNum}"
    })
    
    if (-not $changed) {
        Set-Content $file.FullName -Value $content -Encoding UTF8
    }
}
