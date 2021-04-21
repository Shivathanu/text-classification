import preprocessing

# Simple routine to run a query on a database and print the results:
def doQuery( conn ) :
    cur = conn.cursor()
    cur.execute( """
            SELECT id, summary
            FROM promessa_jira.jira_issues_data;
        """
    )

    for id, summary in cur.fetchall() :
        print('id, summary::', id, summary)
        query = """
            UPDATE promessa_jira.jira_issues_data
            SET clean_summary = "%s"
            WHERE id = %d;
        """
        summary = summary.replace("\"", "\"\"")
        #print(query % (str(preprocessing.text_preprocessing(summary)), id))
        cur.execute(query % (preprocessing.text_preprocessing(summary), id))


print( "Using pymysql:" )
import pymysql
myConnection = pymysql.connect( host="127.0.0.1", user="root", passwd="9840876156", db="promessa_jira" )
doQuery( myConnection )
myConnection.commit()
myConnection.close()
