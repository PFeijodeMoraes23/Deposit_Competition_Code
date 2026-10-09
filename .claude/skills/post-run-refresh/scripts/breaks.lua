local specials = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$",
  ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%", ["_"] = "\\_",
  ["~"] = "\\textasciitilde{}", ["^"] = "\\textasciicircum{}",
}
local function escape(s) return (s:gsub("[\\{}$&#%%_~^]", specials)) end

function Code(el)
  if not FORMAT:match("latex") then return nil end
  local t = escape(el.text)
  t = t:gsub("%-%-", "-{}-")
  t = t:gsub("\\_", "\\_\\allowbreak{}")
  t = t:gsub("([/:=,])", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{" .. t .. "}")
end

function Str(el)
  if not FORMAT:match("latex") then return nil end
  local n = utf8.len(el.text)
  if n == nil or n <= 28 then return nil end
  local t = escape(el.text)
  t = t:gsub("(%l)(%u)", "%1\\allowbreak{}%2")
  t = t:gsub("([/:])", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", t)
end
