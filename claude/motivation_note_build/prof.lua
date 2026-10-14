-- LaTeX output of the professor-facing note: each image paragraph becomes a figure that keeps
-- its place ([H]), drawn from the PDF twin of the PNG (copied next to the .tex), with the alt
-- text as an unnumbered caption whose "Figure N." label is set in bold.
local specials = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$",
  ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%", ["_"] = "\\_",
  ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}",
}
local function escape(s) return (s:gsub("[\\{}$&#%%_~^]", specials)) end

local function pdf_name(src)
  local base = src:match("([^/\\]+)$")
  return (base:gsub("%.png$", ".pdf"))
end

function Para(el)
  if not FORMAT:match("latex") then return nil end
  if #el.content == 1 and el.content[1].t == "Image" then
    local img = el.content[1]
    local cap = pandoc.utils.stringify(img.caption)
    local label, rest = cap:match("^(Figure%s+%S+%.)%s*(.*)$")
    local body
    if label then
      body = "\\textbf{" .. escape(label) .. "} " .. escape(rest)
    else
      body = escape(cap)
    end
    return pandoc.RawBlock("latex",
      "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.78\\linewidth]{" ..
      pdf_name(img.src) .. "}\n\\caption*{" .. body .. "}\n\\end{figure}")
  end
end
