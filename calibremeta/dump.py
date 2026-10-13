import sqlite3, json, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
db.row_factory = sqlite3.Row
q = lambda s, *a: [dict(r) for r in db.execute(s, a)]
books = q("select id,title,sort,author_sort,series_index,pubdate,timestamp,path,has_cover from books")
for b in books:
    i = b['id']
    b['authors'] = [r['name'] for r in q("select a.name from books_authors_link l join authors a on a.id=l.author where l.book=? order by l.id", i)]
    s = q("select s.name from books_series_link l join series s on s.id=l.series where l.book=?", i)
    b['series'] = s[0]['name'] if s else None
    b['tags'] = [r['name'] for r in q("select t.name from books_tags_link l join tags t on t.id=l.tag where l.book=?", i)]
    p = q("select p.name from books_publishers_link l join publishers p on p.id=l.publisher where l.book=?", i)
    b['publisher'] = p[0]['name'] if p else None
    b['langs'] = [r['lang_code'] for r in q("select g.lang_code from books_languages_link l join languages g on g.id=l.lang_code where l.book=?", i)]
    b['ids'] = {r['type']: r['val'] for r in q("select type,val from identifiers where book=?", i)}
    b['formats'] = [r['format'] for r in q("select format from data where book=?", i)]
    c = q("select text from comments where book=?", i)
    b['comments'] = c[0]['text'] if c else None
json.dump(books, open(sys.argv[2]+'/books.json','w'), indent=1, ensure_ascii=False)
json.dump(q("select id,name,sort,link from authors"), open(sys.argv[2]+'/authors.json','w'), indent=1, ensure_ascii=False)
json.dump(q("select name, count(*) n from tags t join books_tags_link l on l.tag=t.id group by t.id order by n desc"), sys.stdout, ensure_ascii=False)
print()
json.dump(q("select name, count(*) n from publishers t join books_publishers_link l on l.publisher=t.id group by t.id order by n desc"), sys.stdout, ensure_ascii=False)
